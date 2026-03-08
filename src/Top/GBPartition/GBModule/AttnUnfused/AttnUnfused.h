#ifndef __ATTN_UNFUSED__
#define __ATTN_UNFUSED__

#include <ac_math.h>
#include <nvhls_module.h>
#include <systemc.h>

#include "AttentionSpec.h"
#include "AttnDatapath.h"

/**
 * Unfused Attention: QK^T, Softmax, and V-multiply as 3 separate stages.
 *
 * Stage 1 (QK^T): For each row i, compute dot(Q[i], K[j])/sqrt(d) for all j,
 *   pack into ceil(N/16) int8 vectors, write score row to GBCore.
 * Stage 2 (Softmax): For each row, read score vectors from GBCore,
 *   compute softmax (max/exp/sum/norm) in 16-element chunks (NMP pattern),
 *   write normalized weights back to GBCore.
 * Stage 3 (V-mul): For each row, read weight vectors from GBCore,
 *   for each j read V[j], accumulate weight[j]*V[j], write output.
 *
 * Maximum GBCore data movement — baseline for comparison.
 * Supports N up to kMaxSeqLen (64).
 */
class AttnUnfused : public match::Module {
  static const int kDebugLevel = 3;
  SC_HAS_PROCESS(AttnUnfused);

public:
  Connections::In<spec::Axi::SubordinateToRVA::Write> rva_in;
  Connections::Out<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::In<bool> start;
  Connections::Out<bool> done;
  Connections::Out<spec::GB::Large::DataReq> large_req;
  Connections::In<spec::GB::Large::DataRsp<1>> large_rsp;

  enum FSM {
    IDLE,
    // Stage 1: QK^T
    QKT_ROW_START,
    QKT_READ_Q_REQ, QKT_READ_Q_RSP,
    QKT_READ_K_REQ, QKT_READ_K_RSP,
    QKT_DOT_SCALE,
    QKT_PACK_VEC,
    QKT_WRITE_VEC,
    QKT_NEXT_ROW,
    // Stage 2: Softmax (NMP-style, 16-element chunks)
    SM_ROW_START,
    SM_READ_REQ, SM_READ_RSP,
    SM_UNPACK,        // unpack + store per-chunk max (no feedback)
    SM_FIND_MAX,      // find global max from chunk_maxes (like NMP SOFTMAX_MAX)
    SM_EXP_SUM,       // exp + partial sum per 16-element chunk (like NMP SOFTMAX_EXP+SUM)
    SM_RECIPROCAL,    // compute 1/sum (like end of NMP SOFTMAX_SUM)
    SM_NORM_PACK,     // normalize + pack per 16-element chunk (like NMP SOFTMAX_NORM)
    SM_WRITE_VEC,
    SM_NEXT_ROW,
    // Stage 3: V multiply (multi-vector weights)
    VM_ROW_START,
    VM_READ_W_REQ, VM_READ_W_RSP, VM_UNPACK_W,
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

  // Score buffer for full row (up to kMaxSeqLen)
  spec::Attention::FixedType score_buf[spec::Attention::kMaxSeqLen];
  NVUINT8 col_counter;
  spec::VectorType write_data;
  NVUINT8 vec_idx;  // multi-vector index for chunk loops

  // Softmax state (following NMP pattern)
  spec::Attention::FixedType max_value;
  // Per-chunk maxes to avoid feedback on max_value (max 4 chunks for kMaxSeqLen=64)
  spec::Attention::FixedType chunk_maxes[spec::Attention::kMaxSeqLen / spec::kVectorSize];
  spec::Attention::UnsignedFixedType exp_values[spec::Attention::kMaxSeqLen];
  spec::Attention::UnsignedAccumType sum_exp;
  spec::Attention::AccumType sum_exp_reciprocal;

  // V-multiply state
  spec::Attention::UnsignedFixedType all_weights[spec::Attention::kMaxSeqLen];
  spec::Attention::AccumType v_accum[spec::kVectorSize];
  spec::Attention::FixedType v_fixed[spec::kVectorSize];
  NVUINT8 vm_col_counter;

  AttnUnfused(sc_module_name nm) :
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
    vec_idx = 0;
    vm_col_counter = 0;
    max_value = spec::kAttentionWordMin;
    sum_exp = 0;
    sum_exp_reciprocal = 0;
#pragma hls_unroll yes
    for (int i = 0; i < spec::kVectorSize; i++) {
      v_accum[i] = 0;
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

  void PushReadReq(NVUINT3 mem_idx, NVUINT8 vec_idx_arg, NVUINT16 ts_idx) {
    large_req_reg.is_write = 0;
    large_req_reg.memory_index = mem_idx;
    large_req_reg.vector_index = vec_idx_arg;
    large_req_reg.timestep_index = ts_idx;
    large_req.Push(large_req_reg);
    attn_config.gb_read_count++;
  }

  void PushWriteReq(NVUINT3 mem_idx, NVUINT8 vec_idx_arg, NVUINT16 ts_idx,
                    const spec::VectorType& data) {
    large_req_reg.is_write = 1;
    large_req_reg.memory_index = mem_idx;
    large_req_reg.vector_index = vec_idx_arg;
    large_req_reg.timestep_index = ts_idx;
    large_req_reg.write_data = data;
    large_req.Push(large_req_reg);
    attn_config.gb_write_count++;
  }

  // =========================================================================
  // Softmax helper functions (following NMP pattern, 16-element operations)
  // =========================================================================

  // Find max within one 16-element chunk, store in chunk_maxes (no max_value feedback)
  void ComputeChunkMax(NVUINT8 base, NVUINT8 chunk_idx) {
    spec::Attention::FixedType vec_max = spec::kAttentionWordMin;
#pragma hls_unroll yes
    for (int i = 0; i < spec::kVectorSize; i++) {
      if ((base + i) < attn_config.seq_len) {
        if (score_buf[base + i] > vec_max) vec_max = score_buf[base + i];
      }
    }
    chunk_maxes[chunk_idx] = vec_max;
  }

  // Like NMP::ComputeSoftmaxMax — find global max from chunk_maxes (write-only to max_value)
  void ComputeGlobalMax(NVUINT8 num_vecs) {
    max_value = spec::kAttentionWordMin;
#pragma hls_unroll yes
    for (int t = 0; t < spec::Attention::kMaxSeqLen / spec::kVectorSize; t++) {
      if (t < num_vecs) {
        if (chunk_maxes[t] > max_value) max_value = chunk_maxes[t];
      }
    }
  }

  // Like NMP::ComputeSoftmaxExp + ComputeSoftmaxSum — exp and partial sum for 16-element chunk
  void ComputeChunkExpSum(NVUINT8 base) {
    spec::Attention::UnsignedAccumType partial_sum = 0;
#pragma hls_unroll yes
    for (int i = 0; i < spec::kVectorSize; i++) {
      if ((base + i) < attn_config.seq_len) {
        spec::Attention::FixedType shifted = score_buf[base + i] - max_value;
        exp_values[base + i] = ac_math::ac_exp_pwl<spec::Attention::UnsignedFixedType>(shifted);
        partial_sum += exp_values[base + i];
      }
    }
    sum_exp += partial_sum;
  }

  // Like NMP::ComputeSoftmaxNormalize + ConvertOutputToInt — normalize and pack 16-element chunk
  void ComputeChunkNormPack(NVUINT8 base) {
    spec::VectorType pack_vec = 0;
#pragma hls_unroll yes
    for (int i = 0; i < spec::kVectorSize; i++) {
      if ((base + i) < attn_config.seq_len) {
        spec::Attention::FixedType normalized = exp_values[base + i] * sum_exp_reciprocal;
        spec::NMP::InputFixedType out_tmp = ConvertToNmpOutputType(normalized);
        pack_vec[i] = nvhls::get_slc<spec::kIntWordWidth>(out_tmp, 0);
      }
    }
    write_data = pack_vec;
  }

  void RunFSM() {
    switch (state) {
      case IDLE: ResetCompute(); break;

      // === Stage 1: QK^T ===
      case QKT_ROW_START: {
        col_counter = 0;
        vec_idx = 0;
        break;
      }
      case QKT_READ_Q_REQ: PushReadReq(attn_config.q_memory_index, 0, attn_config.row_counter); break;
      case QKT_READ_Q_RSP: break;
      case QKT_READ_K_REQ: PushReadReq(attn_config.k_memory_index, 0, col_counter); break;
      case QKT_READ_K_RSP: break;
      case QKT_DOT_SCALE: {
        spec::Attention::FixedType logit;
        AttnDotProductScale(q_row_reg, large_rsp_reg.read_vector[0], logit);
        score_buf[col_counter] = logit;
        col_counter++;
        break;
      }
      case QKT_PACK_VEC: {
        // Pack vec_idx-th vector from score_buf (16 elements)
        NVUINT8 base = vec_idx * spec::kVectorSize;
        spec::VectorType pack_vec = 0;
#pragma hls_unroll yes
        for (int k = 0; k < spec::kVectorSize; k++) {
          if ((base + k) < attn_config.seq_len) {
            spec::NMP::InputFixedType out_tmp = ConvertToNmpOutputType(score_buf[base + k]);
            pack_vec[k] = nvhls::get_slc<spec::kIntWordWidth>(out_tmp, 0);
          }
        }
        write_data = pack_vec;
        break;
      }
      case QKT_WRITE_VEC: {
        PushWriteReq(attn_config.o_memory_index, vec_idx, attn_config.row_counter, write_data);
        vec_idx++;
        break;
      }
      case QKT_NEXT_ROW: break;

      // === Stage 2: Softmax (NMP-style, 16-element chunks) ===
      case SM_ROW_START: {
        vec_idx = 0;
        max_value = spec::kAttentionWordMin;
        sum_exp = 0;
        break;
      }
      case SM_READ_REQ: PushReadReq(attn_config.o_memory_index, vec_idx, attn_config.row_counter); break;
      case SM_READ_RSP: break;
      case SM_UNPACK: {
        // Unpack vec_idx-th vector into score_buf + store per-chunk max
        NVUINT8 base = vec_idx * spec::kVectorSize;
#pragma hls_unroll yes
        for (int k = 0; k < spec::kVectorSize; k++) {
          if ((base + k) < attn_config.seq_len) {
            spec::ScalarType tmp = large_rsp_reg.read_vector[0][k];
            NVINTW(spec::kIntWordWidth) signed_input = (NVINTW(spec::kIntWordWidth))tmp;
            spec::Attention::InputFixedType in_fixed = 0;
            in_fixed.set_slc(0, signed_input);
            score_buf[base + k] = ConvertFromNmpInputType(in_fixed);
          }
        }
        // Store per-chunk max (no max_value feedback)
        ComputeChunkMax(base, vec_idx);
        vec_idx++;
        break;
      }
      case SM_FIND_MAX: {
        // Find global max from chunk maxes (like NMP ComputeSoftmaxMax)
        NVUINT8 num_vecs = (attn_config.seq_len + spec::kVectorSize - 1) / spec::kVectorSize;
        ComputeGlobalMax(num_vecs);
        vec_idx = 0;
        break;
      }
      case SM_EXP_SUM: {
        // Compute exp and partial sum for 16-element chunk vec_idx
        // (like NMP ComputeSoftmaxExp + ComputeSoftmaxSum)
        NVUINT8 base = vec_idx * spec::kVectorSize;
        ComputeChunkExpSum(base);
        vec_idx++;
        break;
      }
      case SM_RECIPROCAL: {
        // Compute 1/sum_exp (like end of NMP ComputeSoftmaxSum)
        ac_math::ac_reciprocal_pwl(sum_exp, sum_exp_reciprocal);
        vec_idx = 0;
        break;
      }
      case SM_NORM_PACK: {
        // Normalize and pack 16-element chunk (like NMP ComputeSoftmaxNormalize)
        NVUINT8 base = vec_idx * spec::kVectorSize;
        ComputeChunkNormPack(base);
        break;
      }
      case SM_WRITE_VEC: {
        PushWriteReq(attn_config.o_memory_index, vec_idx, attn_config.row_counter, write_data);
        vec_idx++;
        break;
      }
      case SM_NEXT_ROW: break;

      // === Stage 3: V multiply (multi-vector weights) ===
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
        // Unpack vec_idx-th weight vector into all_weights (16 elements)
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
    NVUINT8 num_vecs = (attn_config.seq_len + spec::kVectorSize - 1) / spec::kVectorSize;

    switch (state) {
      case IDLE: {
        bool start_reg;
        if (start.PopNB(start_reg)) is_start = attn_config.is_valid && start_reg;
        next_state = is_start ? QKT_ROW_START : IDLE;
        if (is_start) {
          attn_config.ResetCounter();
          attn_config.cycle_count = 0;
          attn_config.gb_read_count = 0;
          attn_config.gb_write_count = 0;
        }
        break;
      }

      // Stage 1: QK^T
      case QKT_ROW_START: next_state = QKT_READ_Q_REQ; break;
      case QKT_READ_Q_REQ: next_state = QKT_READ_Q_RSP; break;
      case QKT_READ_Q_RSP: {
        spec::GB::Large::DataRsp<1> d;
        if (large_rsp.PopNB(d)) { q_row_reg = d.read_vector[0]; next_state = QKT_READ_K_REQ; }
        else next_state = QKT_READ_Q_RSP;
        break;
      }
      case QKT_READ_K_REQ: next_state = QKT_READ_K_RSP; break;
      case QKT_READ_K_RSP: {
        spec::GB::Large::DataRsp<1> d;
        if (large_rsp.PopNB(d)) { large_rsp_reg = d; next_state = QKT_DOT_SCALE; }
        else next_state = QKT_READ_K_RSP;
        break;
      }
      case QKT_DOT_SCALE: {
        next_state = (col_counter >= attn_config.seq_len) ? QKT_PACK_VEC : QKT_READ_K_REQ;
        break;
      }
      case QKT_PACK_VEC: next_state = QKT_WRITE_VEC; break;
      case QKT_WRITE_VEC: {
        next_state = (vec_idx < num_vecs) ? QKT_PACK_VEC : QKT_NEXT_ROW;
        break;
      }
      case QKT_NEXT_ROW: {
        bool row_end = 0;
        attn_config.UpdateRowCounter(row_end);
        next_state = row_end ? SM_ROW_START : QKT_ROW_START;
        break;
      }

      // Stage 2: Softmax (chunk-based)
      case SM_ROW_START: next_state = SM_READ_REQ; break;
      case SM_READ_REQ: next_state = SM_READ_RSP; break;
      case SM_READ_RSP: {
        spec::GB::Large::DataRsp<1> d;
        if (large_rsp.PopNB(d)) { large_rsp_reg = d; next_state = SM_UNPACK; }
        else next_state = SM_READ_RSP;
        break;
      }
      case SM_UNPACK: {
        // After unpack for this chunk, read next or find global max
        next_state = (vec_idx < num_vecs) ? SM_READ_REQ : SM_FIND_MAX;
        break;
      }
      case SM_FIND_MAX: {
        // After finding global max, proceed to exp/sum loop
        next_state = SM_EXP_SUM;
        break;
      }
      case SM_EXP_SUM: {
        // After exp+sum for this chunk, do next chunk or compute reciprocal
        next_state = (vec_idx < num_vecs) ? SM_EXP_SUM : SM_RECIPROCAL;
        break;
      }
      case SM_RECIPROCAL: next_state = SM_NORM_PACK; break;
      case SM_NORM_PACK: next_state = SM_WRITE_VEC; break;
      case SM_WRITE_VEC: {
        next_state = (vec_idx < num_vecs) ? SM_NORM_PACK : SM_NEXT_ROW;
        break;
      }
      case SM_NEXT_ROW: {
        bool row_end = 0;
        attn_config.UpdateRowCounter(row_end);
        next_state = row_end ? VM_ROW_START : SM_ROW_START;
        break;
      }

      // Stage 3: V multiply
      case VM_ROW_START: next_state = VM_READ_W_REQ; break;
      case VM_READ_W_REQ: next_state = VM_READ_W_RSP; break;
      case VM_READ_W_RSP: {
        spec::GB::Large::DataRsp<1> d;
        if (large_rsp.PopNB(d)) { large_rsp_reg = d; next_state = VM_UNPACK_W; }
        else next_state = VM_READ_W_RSP;
        break;
      }
      case VM_UNPACK_W: {
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
