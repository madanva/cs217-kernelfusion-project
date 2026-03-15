#ifndef __ATTN_PARTIAL_FUSED__
#define __ATTN_PARTIAL_FUSED__

#include <ac_math.h>
#include <nvhls_module.h>
#include <systemc.h>

#include "AttentionSpec.h"
#include "AttnDatapath.h"

/**
 * Partially Fused Attention — 2-Thread Dataflow (stall-free).
 *
 * Thread 1 (MemCtrl, II=1): Handles all GBCore reads/writes.
 *   - Reads Q, K vectors and pushes to Compute via channels
 *   - Receives weight vectors from Compute and writes to GBCore
 *   - Reads weight vectors back and V vectors for Stage 2
 *
 * Thread 2 (Compute, II=3): Handles dot products, online softmax,
 *   final normalization, and V-multiply math.
 *
 * Eliminates PopNB stalls that plague the single-thread variant
 * by letting MemCtrl poll at II=1 (1 cycle retry vs 3).
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

  // Internal channels between MemCtrl and Compute
  Connections::Combinational<NVUINT8> cfg_ch;
  Connections::Combinational<spec::VectorType> q_ch;
  Connections::Combinational<spec::VectorType> k_ch;
  Connections::Combinational<spec::VectorType> weight_out_ch;  // Compute→MemCtrl: weight vectors to write
  Connections::Combinational<spec::VectorType> weight_in_ch;   // MemCtrl→Compute: weight vectors read back
  Connections::Combinational<spec::VectorType> v_ch;
  Connections::Combinational<spec::VectorType> output_ch;      // Compute→MemCtrl: final output

  enum MemFSM {
    M_IDLE, M_SEND_CFG, M_ROW_START,
    // Stage 1: read Q+K, receive+write weight vectors
    M_READ_Q_REQ, M_READ_Q_RSP, M_PUSH_Q,
    M_READ_K_REQ, M_READ_K_RSP, M_PUSH_K,
    M_RECV_WEIGHT, M_WRITE_WEIGHT,
    M_NEXT_ROW_S1,
    // Stage 2: read weights + V, receive+write output
    M_S2_ROW_START,
    M_READ_W_REQ, M_READ_W_RSP, M_PUSH_W,
    M_READ_V_REQ, M_READ_V_RSP, M_PUSH_V,
    M_RECV_OUTPUT, M_WRITE_OUTPUT,
    M_NEXT_ROW_S2,
    M_FIN
  };

  enum ComputeFSM {
    C_WAIT_CFG, C_WAIT_Q,
    C_RECV_K,
    C_TILE_MAX, C_TILE_EXP, C_TILE_SUM,
    C_FINAL_NORM, C_SEND_WEIGHT,
    C_NEXT_ROW_S1,
    // Stage 2
    C_RECV_W,
    C_RECV_V,
    C_WRITE_OUT, C_SEND_OUTPUT,
    C_NEXT_ROW_S2
  };

  AttnPartialFused(sc_module_name nm) :
      match::Module(nm),
      rva_in("rva_in"), rva_out("rva_out"),
      start("start"), done("done"),
      large_req("large_req"), large_rsp("large_rsp"),
      cfg_ch("cfg_ch"), q_ch("q_ch"), k_ch("k_ch"),
      weight_out_ch("weight_out_ch"), weight_in_ch("weight_in_ch"),
      v_ch("v_ch"), output_ch("output_ch") {
    SC_THREAD(MemCtrlRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);

    SC_THREAD(ComputeRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  // =========================================================================
  // Thread 1: Memory Controller (II=1)
  // =========================================================================
  void MemCtrlRun() {
    rva_in.Reset(); rva_out.Reset();
    start.Reset(); done.Reset();
    large_req.Reset(); large_rsp.Reset();
    cfg_ch.ResetWrite(); q_ch.ResetWrite(); k_ch.ResetWrite();
    weight_out_ch.ResetRead(); weight_in_ch.ResetWrite();
    v_ch.ResetWrite(); output_ch.ResetRead();

    MemFSM m_state = M_IDLE;
    spec::Attention::AttentionConfig attn_config;
    attn_config.Reset();

    bool is_start = 0;
    bool w_axi_rsp = 0, w_done = 0;
    spec::Axi::SubordinateToRVA::Read rva_out_reg;
    spec::GB::Large::DataReq large_req_reg;
    spec::VectorType mem_data_reg;

    NVUINT8 m_row_counter = 0;
    NVUINT8 m_col_counter = 0;
    NVUINT8 m_vec_idx = 0;
    NVUINT8 m_v_col = 0;

#pragma hls_pipeline_init_interval 1
    while (1) {
      w_axi_rsp = 0;
      w_done = 0;

      spec::Axi::SubordinateToRVA::Write rva_in_reg;
      if (rva_in.PopNB(rva_in_reg)) {
        NVUINT4 tmp = nvhls::get_slc<4>(rva_in_reg.addr, 20);
        NVUINT16 local_index = nvhls::get_slc<16>(rva_in_reg.addr, 4);
        if (rva_in_reg.rw) {
          if (tmp == 0xD) attn_config.ConfigWrite(local_index, rva_in_reg.data);
        } else {
          w_axi_rsp = 1;
          if (tmp == 0xD) attn_config.ConfigRead(local_index, rva_out_reg.data);
        }
      } else {
        NVUINT8 num_vecs = (attn_config.seq_len + spec::kVectorSize - 1) / spec::kVectorSize;

        switch (m_state) {
          case M_IDLE: {
            bool start_reg;
            if (start.PopNB(start_reg)) {
              is_start = attn_config.is_valid && start_reg;
            }
            if (is_start) {
              attn_config.ResetCounter();
              attn_config.cycle_count = 0;
              attn_config.gb_read_count = 0;
              attn_config.gb_write_count = 0;
              m_row_counter = 0;
              m_state = M_SEND_CFG;
            }
            break;
          }
          case M_SEND_CFG: { cfg_ch.Push(attn_config.seq_len); m_state = M_ROW_START; break; }

          // === Stage 1: Read Q+K, receive weight vectors ===
          case M_ROW_START: { m_col_counter = 0; m_vec_idx = 0; m_state = M_READ_Q_REQ; break; }
          case M_READ_Q_REQ: {
            large_req_reg.is_write = 0;
            large_req_reg.memory_index = attn_config.q_memory_index;
            large_req_reg.vector_index = 0;
            large_req_reg.timestep_index = m_row_counter;
            large_req.Push(large_req_reg);
            attn_config.gb_read_count++;
            m_state = M_READ_Q_RSP;
            break;
          }
          case M_READ_Q_RSP: {
            spec::GB::Large::DataRsp<1> rsp;
            if (large_rsp.PopNB(rsp)) { mem_data_reg = rsp.read_vector[0]; m_state = M_PUSH_Q; }
            break;
          }
          case M_PUSH_Q: { q_ch.Push(mem_data_reg); m_state = M_READ_K_REQ; break; }

          case M_READ_K_REQ: {
            large_req_reg.is_write = 0;
            large_req_reg.memory_index = attn_config.k_memory_index;
            large_req_reg.vector_index = 0;
            large_req_reg.timestep_index = m_col_counter;
            large_req.Push(large_req_reg);
            attn_config.gb_read_count++;
            m_state = M_READ_K_RSP;
            break;
          }
          case M_READ_K_RSP: {
            spec::GB::Large::DataRsp<1> rsp;
            if (large_rsp.PopNB(rsp)) { mem_data_reg = rsp.read_vector[0]; m_state = M_PUSH_K; }
            break;
          }
          case M_PUSH_K: {
            k_ch.Push(mem_data_reg);
            m_col_counter++;
            if (m_col_counter >= attn_config.seq_len) {
              m_state = M_RECV_WEIGHT;  // All K sent, wait for weight vectors
            } else {
              m_state = M_READ_K_REQ;
            }
            break;
          }

          case M_RECV_WEIGHT: {
            spec::VectorType w_data;
            if (weight_out_ch.PopNB(w_data)) { mem_data_reg = w_data; m_state = M_WRITE_WEIGHT; }
            break;
          }
          case M_WRITE_WEIGHT: {
            large_req_reg.is_write = 1;
            large_req_reg.memory_index = attn_config.o_memory_index;
            large_req_reg.vector_index = m_vec_idx;
            large_req_reg.timestep_index = m_row_counter;
            large_req_reg.write_data = mem_data_reg;
            large_req.Push(large_req_reg);
            attn_config.gb_write_count++;
            m_vec_idx++;
            m_state = (m_vec_idx < num_vecs) ? M_RECV_WEIGHT : M_NEXT_ROW_S1;
            break;
          }

          case M_NEXT_ROW_S1: {
            m_row_counter++;
            m_state = (m_row_counter >= attn_config.seq_len) ? M_S2_ROW_START : M_ROW_START;
            if (m_row_counter >= attn_config.seq_len) m_row_counter = 0;
            break;
          }

          // === Stage 2: Read weights + V, receive output ===
          case M_S2_ROW_START: { m_vec_idx = 0; m_v_col = 0; m_state = M_READ_W_REQ; break; }
          case M_READ_W_REQ: {
            large_req_reg.is_write = 0;
            large_req_reg.memory_index = attn_config.o_memory_index;
            large_req_reg.vector_index = m_vec_idx;
            large_req_reg.timestep_index = m_row_counter;
            large_req.Push(large_req_reg);
            attn_config.gb_read_count++;
            m_state = M_READ_W_RSP;
            break;
          }
          case M_READ_W_RSP: {
            spec::GB::Large::DataRsp<1> rsp;
            if (large_rsp.PopNB(rsp)) { mem_data_reg = rsp.read_vector[0]; m_state = M_PUSH_W; }
            break;
          }
          case M_PUSH_W: {
            weight_in_ch.Push(mem_data_reg);
            m_vec_idx++;
            m_state = (m_vec_idx < num_vecs) ? M_READ_W_REQ : M_READ_V_REQ;
            break;
          }

          case M_READ_V_REQ: {
            large_req_reg.is_write = 0;
            large_req_reg.memory_index = attn_config.v_memory_index;
            large_req_reg.vector_index = 0;
            large_req_reg.timestep_index = m_v_col;
            large_req.Push(large_req_reg);
            attn_config.gb_read_count++;
            m_state = M_READ_V_RSP;
            break;
          }
          case M_READ_V_RSP: {
            spec::GB::Large::DataRsp<1> rsp;
            if (large_rsp.PopNB(rsp)) { mem_data_reg = rsp.read_vector[0]; m_state = M_PUSH_V; }
            break;
          }
          case M_PUSH_V: {
            v_ch.Push(mem_data_reg);
            m_v_col++;
            if (m_v_col >= attn_config.seq_len) {
              m_state = M_RECV_OUTPUT;
            } else {
              m_state = M_READ_V_REQ;
            }
            break;
          }

          case M_RECV_OUTPUT: {
            spec::VectorType out_data;
            if (output_ch.PopNB(out_data)) { mem_data_reg = out_data; m_state = M_WRITE_OUTPUT; }
            break;
          }
          case M_WRITE_OUTPUT: {
            large_req_reg.is_write = 1;
            large_req_reg.memory_index = attn_config.o_memory_index;
            large_req_reg.vector_index = 0;
            large_req_reg.timestep_index = m_row_counter;
            large_req_reg.write_data = mem_data_reg;
            large_req.Push(large_req_reg);
            attn_config.gb_write_count++;
            m_state = M_NEXT_ROW_S2;
            break;
          }

          case M_NEXT_ROW_S2: {
            m_row_counter++;
            m_state = (m_row_counter >= attn_config.seq_len) ? M_FIN : M_S2_ROW_START;
            break;
          }

          case M_FIN: { is_start = 0; w_done = 1; m_state = M_IDLE; break; }
          default: m_state = M_IDLE; break;
        }
      }

      if (w_axi_rsp) rva_out.Push(rva_out_reg);
      if (w_done) done.Push(1);
      if (m_state != M_IDLE) attn_config.cycle_count++;
      wait();
    }
  }

  // =========================================================================
  // Thread 2: Compute (II=3)
  // =========================================================================
  void ComputeRun() {
    cfg_ch.ResetRead(); q_ch.ResetRead(); k_ch.ResetRead();
    weight_out_ch.ResetWrite(); weight_in_ch.ResetRead();
    v_ch.ResetRead(); output_ch.ResetWrite();

    ComputeFSM c_state = C_WAIT_CFG;

    NVUINT8 seq_len = 0;
    NVUINT8 row_counter = 0;
    NVUINT8 col_counter = 0;
    NVUINT8 tile_count = 0;
    NVUINT8 vec_write_idx = 0;
    NVUINT8 vec_idx = 0;
    NVUINT8 vm_col_counter = 0;

    spec::VectorType q_row_reg;
    spec::VectorType write_data;

    spec::Attention::FixedType all_logits[spec::Attention::kMaxSeqLen];
    spec::Attention::FixedType tile_logits[spec::Attention::kTileSize];
    spec::Attention::FixedType running_max;
    spec::Attention::UnsignedAccumType running_sum;

    spec::Attention::AccumType v_accum[spec::kVectorSize];
    spec::Attention::UnsignedFixedType all_weights[spec::Attention::kMaxSeqLen];
    spec::Attention::FixedType v_fixed[spec::kVectorSize];

    running_max = spec::kAttentionWordMin;
    running_sum = 0;
#pragma hls_unroll yes
    for (int i = 0; i < spec::kVectorSize; i++) { v_accum[i] = 0; }

#pragma hls_pipeline_init_interval 3
    while (1) {
      switch (c_state) {
        case C_WAIT_CFG: {
          NVUINT8 cfg;
          if (cfg_ch.PopNB(cfg)) { seq_len = cfg; row_counter = 0; c_state = C_WAIT_Q; }
          break;
        }

        case C_WAIT_Q: {
          spec::VectorType q_data;
          if (q_ch.PopNB(q_data)) {
            q_row_reg = q_data;
            col_counter = 0;
            tile_count = 0;
            vec_write_idx = 0;
            running_max = spec::kAttentionWordMin;
            running_sum = 0;
            c_state = C_RECV_K;
          }
          break;
        }

        case C_RECV_K: {
          spec::VectorType k_data;
          if (k_ch.PopNB(k_data)) {
            spec::Attention::FixedType logit;
            AttnDotProductScale(q_row_reg, k_data, logit);
            all_logits[col_counter] = logit;
            tile_logits[tile_count] = logit;
            tile_count++;
            col_counter++;
            bool tile_full = (tile_count >= spec::Attention::kTileSize);
            bool row_done = (col_counter >= seq_len);
            c_state = (tile_full || row_done) ? C_TILE_MAX : C_RECV_K;
          }
          break;
        }

        case C_TILE_MAX: {
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
          running_max = new_max;
          c_state = C_TILE_EXP;
          break;
        }

        case C_TILE_EXP: {
          spec::Attention::UnsignedAccumType tile_sum = 0;
#pragma hls_unroll yes
#pragma cluster addtree
#pragma cluster_type both
          for (int i = 0; i < spec::Attention::kTileSize; i++) {
            if (i < tile_count) {
              spec::Attention::FixedType shifted = tile_logits[i] - running_max;
              spec::Attention::UnsignedFixedType e =
                  ac_math::ac_exp_pwl<spec::Attention::UnsignedFixedType>(shifted);
              tile_sum += e;
            }
          }
          running_sum += tile_sum;
          c_state = C_TILE_SUM;
          break;
        }

        case C_TILE_SUM: {
          tile_count = 0;
          c_state = (col_counter >= seq_len) ? C_FINAL_NORM : C_RECV_K;
          break;
        }

        case C_FINAL_NORM: {
          // Compute normalized weights for vec_write_idx-th vector
          NVUINT8 base = vec_write_idx * spec::kVectorSize;
          spec::Attention::AccumType sum_recip;
          ac_math::ac_reciprocal_pwl(running_sum, sum_recip);

          spec::VectorType pack_vec = 0;
#pragma hls_unroll yes
          for (int k = 0; k < spec::kVectorSize; k++) {
            if ((base + k) < seq_len) {
              spec::Attention::FixedType shifted = all_logits[base + k] - running_max;
              spec::Attention::UnsignedFixedType e =
                  ac_math::ac_exp_pwl<spec::Attention::UnsignedFixedType>(shifted);
              spec::Attention::FixedType normalized = e * sum_recip;
              spec::NMP::InputFixedType out_tmp = ConvertToNmpOutputType(normalized);
              pack_vec[k] = nvhls::get_slc<spec::kIntWordWidth>(out_tmp, 0);
            }
          }
          write_data = pack_vec;
          c_state = C_SEND_WEIGHT;
          break;
        }

        case C_SEND_WEIGHT: {
          weight_out_ch.Push(write_data);
          vec_write_idx++;
          NVUINT8 num_vecs = (seq_len + spec::kVectorSize - 1) / spec::kVectorSize;
          c_state = (vec_write_idx < num_vecs) ? C_FINAL_NORM : C_NEXT_ROW_S1;
          break;
        }

        case C_NEXT_ROW_S1: {
          row_counter++;
          c_state = (row_counter >= seq_len) ? C_RECV_W : C_WAIT_Q;
          if (row_counter >= seq_len) { row_counter = 0; vec_idx = 0; }
          break;
        }

        // === Stage 2: V multiply ===
        case C_RECV_W: {
          spec::VectorType w_data;
          if (weight_in_ch.PopNB(w_data)) {
            NVUINT8 base = vec_idx * spec::kVectorSize;
#pragma hls_unroll yes
            for (int k = 0; k < spec::kVectorSize; k++) {
              if ((base + k) < seq_len) {
                spec::ScalarType tmp = w_data[k];
                NVINTW(spec::kIntWordWidth) signed_input = (NVINTW(spec::kIntWordWidth))tmp;
                spec::Attention::InputFixedType in_fixed = 0;
                in_fixed.set_slc(0, signed_input);
                all_weights[base + k] = ConvertFromNmpInputType(in_fixed);
              }
            }
            vec_idx++;
            NVUINT8 num_vecs = (seq_len + spec::kVectorSize - 1) / spec::kVectorSize;
            c_state = (vec_idx < num_vecs) ? C_RECV_W : C_RECV_V;
            if (vec_idx >= num_vecs) {
              vm_col_counter = 0;
#pragma hls_unroll yes
              for (int k = 0; k < spec::kVectorSize; k++) v_accum[k] = 0;
            }
          }
          break;
        }

        case C_RECV_V: {
          spec::VectorType v_data;
          if (v_ch.PopNB(v_data)) {
            AttnConvertInputToFixed(v_data, v_fixed);
            AttnWeightedAccum(all_weights[vm_col_counter], v_fixed, v_accum);
            vm_col_counter++;
            c_state = (vm_col_counter >= seq_len) ? C_WRITE_OUT : C_RECV_V;
          }
          break;
        }

        case C_WRITE_OUT: {
#pragma hls_unroll yes
          for (int k = 0; k < spec::kVectorSize; k++) {
            spec::Attention::FixedType val = v_accum[k];
            spec::NMP::InputFixedType out_tmp = ConvertToNmpOutputType(val);
            write_data[k] = nvhls::get_slc<spec::kIntWordWidth>(out_tmp, 0);
          }
          c_state = C_SEND_OUTPUT;
          break;
        }

        case C_SEND_OUTPUT: {
          output_ch.Push(write_data);
          row_counter++;
          if (row_counter >= seq_len) {
            c_state = C_WAIT_CFG;
          } else {
            vec_idx = 0;
            c_state = C_RECV_W;
          }
          break;
        }

        default: c_state = C_WAIT_CFG; break;
      }
      wait();
    }
  }
};

#endif
