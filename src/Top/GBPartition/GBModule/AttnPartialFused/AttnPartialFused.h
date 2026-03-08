#ifndef __ATTN_PARTIAL_FUSED__
#define __ATTN_PARTIAL_FUSED__

#include <ac_math.h>
#include <nvhls_module.h>
#include <systemc.h>

#include "AttentionSpec.h"
#include "AttnDatapath.h"

/**
 * @brief Partially Fused Attention: QK^T + online softmax fused, V-multiply separate.
 *
 * For each row i:
 *   1. Read Q[i], compute all dot products with K[0..N-1], apply online softmax
 *   2. Write normalized attention weights row to GBCore
 * Then for each row i:
 *   3. Read attention weights, read V vectors, MAC to produce output
 *
 * Eliminates QK^T intermediate (N×N raw scores). Still writes/reads attention weights.
 */
class AttnPartialFused : public match::Module {
  static const int kDebugLevel = 3;
  SC_HAS_PROCESS(AttnPartialFused);

public:
  Connections::In<spec::Axi::SubordinateToRVA::Write> rva_in;
  Connections::Out<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::In<bool> start;
  Connections::Out<bool> done;
  Connections::Out<spec::GB::Large::DataReq> large_req;
  Connections::In<spec::GB::Large::DataRsp<1>> large_rsp;

  enum FSM {
    IDLE,
    // Stage 1: Fused QK^T + softmax
    QKS_ROW_START,
    QKS_READ_Q_REQ, QKS_READ_Q_RSP,
    QKS_READ_K_REQ, QKS_READ_K_RSP,
    QKS_DOT_SCALE,
    QKS_TILE_SOFTMAX,
    QKS_NEXT_TILE,
    QKS_FINAL_NORM,
    QKS_WRITE_VEC,
    QKS_NEXT_ROW,
    // Stage 2: V multiply (batch weight reads, then sequential V reads)
    VM_ROW_START,
    VM_READ_W_REQ, VM_READ_W_RSP,
    VM_UNPACK_W,
    VM_READ_V_REQ, VM_READ_V_RSP,
    VM_MAC,
    VM_WRITE_OUTPUT,
    VM_NEXT_ROW,
    FIN
  };
  FSM state, next_state;

  bool is_start;
  spec::Attention::AttentionConfig attn_config;
  bool w_axi_rsp, w_done;
  spec::Axi::SubordinateToRVA::Read rva_out_reg;

  spec::GB::Large::DataReq large_req_reg;
  spec::GB::Large::DataRsp<1> large_rsp_reg;

  // Q row register
  spec::VectorType q_row_reg;

  // Online softmax state (across all N columns)
  spec::Attention::FixedType all_logits[spec::Attention::kMaxSeqLen];
  NVUINT8 col_counter;
  spec::Attention::FixedType running_max;
  spec::Attention::UnsignedAccumType running_sum;

  // Tile tracking
  spec::Attention::FixedType tile_logits[spec::Attention::kTileSize];
  NVUINT8 tile_count;

  // Normalized weights for writing
  spec::VectorType write_data;
  NVUINT8 vec_write_idx;

  // V-multiply state (batch pattern: read all weights first, then sequential V reads)
  spec::Attention::AccumType v_accum[spec::kVectorSize];
  spec::Attention::UnsignedFixedType all_weights[spec::Attention::kMaxSeqLen];
  NVUINT8 vec_idx;        // which weight vector being read
  NVUINT8 vm_col_counter; // which V column (sequential 0..N-1)
  spec::Attention::FixedType v_fixed[spec::kVectorSize];

  AttnPartialFused(sc_module_name nm) :
      match::Module(nm),
      rva_in("rva_in"), rva_out("rva_out"),
      start("start"), done("done"),
      large_req("large_req"), large_rsp("large_rsp") {
    SC_THREAD(AttnRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void Reset() {
    state = IDLE;
    is_start = 0;
    w_axi_rsp = 0;
    w_done = 0;
    attn_config.Reset();
    rva_in.Reset(); rva_out.Reset();
    start.Reset(); done.Reset();
    large_req.Reset(); large_rsp.Reset();
    ResetCompute();
  }

  void ResetCompute() {
    col_counter = 0;
    tile_count = 0;
    running_max = spec::kAttentionWordMin;
    running_sum = 0;
    vec_write_idx = 0;
    vec_idx = 0;
    vm_col_counter = 0;
#pragma hls_unroll yes
    for (int k = 0; k < spec::kVectorSize; k++) {
      v_accum[k] = 0;
    }
  }

  void DecodeAxiWrite(const spec::Axi::SubordinateToRVA::Write& rva_in_reg) {
    NVUINT4 tmp = nvhls::get_slc<4>(rva_in_reg.addr, 20);
    NVUINT16 local_index = nvhls::get_slc<16>(rva_in_reg.addr, 4);
    if (tmp == 0xD) attn_config.ConfigWrite(local_index, rva_in_reg.data);
  }

  void DecodeAxiRead(const spec::Axi::SubordinateToRVA::Write& rva_in_reg) {
    NVUINT4 tmp = nvhls::get_slc<4>(rva_in_reg.addr, 20);
    NVUINT16 local_index = nvhls::get_slc<16>(rva_in_reg.addr, 4);
    w_axi_rsp = 1;
    if (tmp == 0xD) attn_config.ConfigRead(local_index, rva_out_reg.data);
  }

  void PushReadReq(NVUINT3 mem_idx, NVUINT8 vec_idx, NVUINT16 ts_idx) {
    large_req_reg.is_write = 0;
    large_req_reg.memory_index = mem_idx;
    large_req_reg.vector_index = vec_idx;
    large_req_reg.timestep_index = ts_idx;
    large_req.Push(large_req_reg);
    attn_config.gb_read_count++;
  }

  void PushWriteReq(NVUINT3 mem_idx, NVUINT8 vec_idx, NVUINT16 ts_idx,
                    const spec::VectorType& data) {
    large_req_reg.is_write = 1;
    large_req_reg.memory_index = mem_idx;
    large_req_reg.vector_index = vec_idx;
    large_req_reg.timestep_index = ts_idx;
    large_req_reg.write_data = data;
    large_req.Push(large_req_reg);
    attn_config.gb_write_count++;
  }

  void TileSoftmaxUpdate() {
    spec::Attention::FixedType tile_max = spec::kAttentionWordMin;
#pragma hls_unroll yes
    for (int i = 0; i < spec::Attention::kTileSize; i++) {
      if (i < tile_count && tile_logits[i] > tile_max) tile_max = tile_logits[i];
    }

    spec::Attention::FixedType new_max = (tile_max > running_max) ? tile_max : running_max;
    spec::Attention::FixedType corr_input = running_max - new_max;
    spec::Attention::UnsignedFixedType correction =
        ac_math::ac_exp_pwl<spec::Attention::UnsignedFixedType>(corr_input);

    running_sum = running_sum * correction;

    spec::Attention::UnsignedAccumType tile_sum = 0;
#pragma hls_unroll yes
#pragma cluster addtree
#pragma cluster_type both
    for (int i = 0; i < spec::Attention::kTileSize; i++) {
      if (i < tile_count) {
        spec::Attention::FixedType shifted = tile_logits[i] - new_max;
        spec::Attention::UnsignedFixedType e =
            ac_math::ac_exp_pwl<spec::Attention::UnsignedFixedType>(shifted);
        tile_sum += e;
      }
    }
    running_sum += tile_sum;
    running_max = new_max;
  }

  // Compute final normalized weights and pack into vectors for writing
  void ComputeFinalWeightsAndPack() {
    NVUINT8 base = vec_write_idx * spec::kVectorSize;
    spec::Attention::AccumType sum_recip;
    ac_math::ac_reciprocal_pwl(running_sum, sum_recip);

    spec::VectorType pack_vec = 0;
#pragma hls_unroll yes
    for (int k = 0; k < spec::kVectorSize; k++) {
      if ((base + k) < attn_config.seq_len) {
        spec::Attention::FixedType shifted = all_logits[base + k] - running_max;
        spec::Attention::UnsignedFixedType e =
            ac_math::ac_exp_pwl<spec::Attention::UnsignedFixedType>(shifted);
        spec::Attention::FixedType normalized = e * sum_recip;
        spec::NMP::InputFixedType out_tmp = ConvertToNmpOutputType(normalized);
        pack_vec[k] = nvhls::get_slc<spec::kIntWordWidth>(out_tmp, 0);
      }
    }
    write_data = pack_vec;
  }

  void RunFSM() {
    switch (state) {
      case IDLE: ResetCompute(); break;

      // Stage 1: Fused QK^T + online softmax
      case QKS_ROW_START: {
        col_counter = 0;
        tile_count = 0;
        vec_write_idx = 0;
        running_max = spec::kAttentionWordMin;
        running_sum = 0;
        break;
      }
      case QKS_READ_Q_REQ: PushReadReq(attn_config.q_memory_index, 0, attn_config.row_counter); break;
      case QKS_READ_Q_RSP: break;
      case QKS_READ_K_REQ: PushReadReq(attn_config.k_memory_index, 0, col_counter); break;
      case QKS_READ_K_RSP: break;
      case QKS_DOT_SCALE: {
        spec::Attention::FixedType logit;
        AttnDotProductScale(q_row_reg, large_rsp_reg.read_vector[0], logit);
        all_logits[col_counter] = logit;
        tile_logits[tile_count] = logit;
        tile_count++;
        col_counter++;
        break;
      }
      case QKS_TILE_SOFTMAX: {
        TileSoftmaxUpdate();
        tile_count = 0;
        break;
      }
      case QKS_NEXT_TILE: break;
      case QKS_FINAL_NORM: {
        ComputeFinalWeightsAndPack();
        break;
      }
      case QKS_WRITE_VEC: {
        PushWriteReq(attn_config.o_memory_index, vec_write_idx, attn_config.row_counter, write_data);
        vec_write_idx++;
        break;
      }
      case QKS_NEXT_ROW: break;

      // Stage 2: V multiply
      case VM_ROW_START: {
        vec_idx = 0;
        vm_col_counter = 0;
#pragma hls_unroll yes
        for (int k = 0; k < spec::kVectorSize; k++) v_accum[k] = 0;
        break;
      }
      case VM_READ_W_REQ: PushReadReq(attn_config.o_memory_index, vec_idx, attn_config.row_counter); break;
      case VM_READ_W_RSP: break;
      case VM_UNPACK_W: {
        // Unpack vec_idx-th weight vector into all_weights (batch read pattern)
        NVUINT8 base = vec_idx * spec::kVectorSize;
#pragma hls_unroll yes
        for (int k = 0; k < spec::kVectorSize; k++) {
          if ((base + k) < attn_config.seq_len) {
            spec::ScalarType tmp = large_rsp_reg.read_vector[0][k];
            NVINTW(spec::kIntWordWidth) signed_input = (NVINTW(spec::kIntWordWidth))tmp;
            spec::Attention::InputFixedType in_fixed = 0;
            in_fixed.set_slc(0, signed_input);
            all_weights[base + k] = ConvertFromNmpInputType(in_fixed);
          }
        }
        vec_idx++;
        break;
      }
      case VM_READ_V_REQ: PushReadReq(attn_config.v_memory_index, 0, vm_col_counter); break;
      case VM_READ_V_RSP: break;
      case VM_MAC: {
        AttnConvertInputToFixed(large_rsp_reg.read_vector[0], v_fixed);
        AttnWeightedAccum(all_weights[vm_col_counter], v_fixed, v_accum);
        vm_col_counter++;
        break;
      }
      case VM_WRITE_OUTPUT: {
#pragma hls_unroll yes
        for (int k = 0; k < spec::kVectorSize; k++) {
          spec::Attention::FixedType val = v_accum[k];
          spec::NMP::InputFixedType out_tmp = ConvertToNmpOutputType(val);
          write_data[k] = nvhls::get_slc<spec::kIntWordWidth>(out_tmp, 0);
        }
        PushWriteReq(attn_config.o_memory_index, 0, attn_config.row_counter, write_data);
        break;
      }
      case VM_NEXT_ROW: break;

      case FIN: { is_start = 0; w_done = 1; break; }
      default: break;
    }
  }

  void UpdateFSM() {
    switch (state) {
      case IDLE: {
        bool start_reg;
        if (start.PopNB(start_reg)) is_start = attn_config.is_valid && start_reg;
        next_state = is_start ? QKS_ROW_START : IDLE;
        if (is_start) {
          attn_config.ResetCounter();
          attn_config.cycle_count = 0;
          attn_config.gb_read_count = 0;
          attn_config.gb_write_count = 0;
        }
        break;
      }

      // Stage 1
      case QKS_ROW_START: next_state = QKS_READ_Q_REQ; break;
      case QKS_READ_Q_REQ: next_state = QKS_READ_Q_RSP; break;
      case QKS_READ_Q_RSP: {
        spec::GB::Large::DataRsp<1> d;
        if (large_rsp.PopNB(d)) { q_row_reg = d.read_vector[0]; next_state = QKS_READ_K_REQ; }
        else next_state = QKS_READ_Q_RSP;
        break;
      }
      case QKS_READ_K_REQ: next_state = QKS_READ_K_RSP; break;
      case QKS_READ_K_RSP: {
        spec::GB::Large::DataRsp<1> d;
        if (large_rsp.PopNB(d)) { large_rsp_reg = d; next_state = QKS_DOT_SCALE; }
        else next_state = QKS_READ_K_RSP;
        break;
      }
      case QKS_DOT_SCALE: {
        bool tile_full = (tile_count >= spec::Attention::kTileSize);
        bool row_done = (col_counter >= attn_config.seq_len);
        if (tile_full || row_done) next_state = QKS_TILE_SOFTMAX;
        else next_state = QKS_READ_K_REQ;
        break;
      }
      case QKS_TILE_SOFTMAX: {
        next_state = (col_counter >= attn_config.seq_len) ? QKS_FINAL_NORM : QKS_READ_K_REQ;
        break;
      }
      case QKS_NEXT_TILE: next_state = QKS_READ_K_REQ; break;
      case QKS_FINAL_NORM: next_state = QKS_WRITE_VEC; break;
      case QKS_WRITE_VEC: {
        NVUINT8 num_vecs = (attn_config.seq_len + spec::kVectorSize - 1) / spec::kVectorSize;
        if (vec_write_idx < num_vecs) {
          // Need to compute next vector's weights
          next_state = QKS_FINAL_NORM;
        } else {
          next_state = QKS_NEXT_ROW;
        }
        break;
      }
      case QKS_NEXT_ROW: {
        bool row_end = 0;
        attn_config.UpdateRowCounter(row_end);
        next_state = row_end ? VM_ROW_START : QKS_ROW_START;
        break;
      }

      // Stage 2: batch weight reads, then sequential V reads (matches unfused pattern)
      case VM_ROW_START: next_state = VM_READ_W_REQ; break;
      case VM_READ_W_REQ: next_state = VM_READ_W_RSP; break;
      case VM_READ_W_RSP: {
        spec::GB::Large::DataRsp<1> d;
        if (large_rsp.PopNB(d)) { large_rsp_reg = d; next_state = VM_UNPACK_W; }
        else next_state = VM_READ_W_RSP;
        break;
      }
      case VM_UNPACK_W: {
        NVUINT8 num_vecs = (attn_config.seq_len + spec::kVectorSize - 1) / spec::kVectorSize;
        next_state = (vec_idx < num_vecs) ? VM_READ_W_REQ : VM_READ_V_REQ;
        break;
      }
      case VM_READ_V_REQ: next_state = VM_READ_V_RSP; break;
      case VM_READ_V_RSP: {
        spec::GB::Large::DataRsp<1> d;
        if (large_rsp.PopNB(d)) { large_rsp_reg = d; next_state = VM_MAC; }
        else next_state = VM_READ_V_RSP;
        break;
      }
      case VM_MAC: {
        next_state = (vm_col_counter >= attn_config.seq_len) ? VM_WRITE_OUTPUT : VM_READ_V_REQ;
        break;
      }
      case VM_WRITE_OUTPUT: next_state = VM_NEXT_ROW; break;
      case VM_NEXT_ROW: {
        bool row_end = 0;
        attn_config.UpdateRowCounter(row_end);
        next_state = row_end ? FIN : VM_ROW_START;
        break;
      }

      case FIN: next_state = IDLE; break;
      default: next_state = IDLE; break;
    }
    state = next_state;
  }

  void AttnRun() {
    Reset();
#pragma hls_pipeline_init_interval 3
    while (1) {
      w_axi_rsp = 0;
      w_done = 0;
      spec::Axi::SubordinateToRVA::Write rva_in_reg;
      if (rva_in.PopNB(rva_in_reg)) {
        if (rva_in_reg.rw) DecodeAxiWrite(rva_in_reg);
        else DecodeAxiRead(rva_in_reg);
      } else {
        RunFSM();
        UpdateFSM();
      }
      if (w_axi_rsp) rva_out.Push(rva_out_reg);
      if (w_done) done.Push(1);
      if (state != IDLE) attn_config.cycle_count++;
      wait();
    }
  }
};

#endif
