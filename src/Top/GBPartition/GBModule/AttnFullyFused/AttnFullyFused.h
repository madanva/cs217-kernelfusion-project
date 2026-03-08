#ifndef __ATTN_FULLY_FUSED__
#define __ATTN_FULLY_FUSED__

#include <ac_math.h>
#include <nvhls_module.h>
#include <systemc.h>

#include "AttentionSpec.h"
#include "AttnDatapath.h"

/**
 * Fully Fused Attention: 2-thread dataflow architecture.
 *
 * Thread 1 (MemCtrl): Handles all GBCore reads/writes, AXI config, and
 *   sequencing. Lightweight FSM — no heavy compute.
 *
 * Thread 2 (Compute): Handles dot products, online softmax, V accumulation,
 *   and final normalization. No GBCore access — all data via internal channels.
 *
 * This split reduces the HLS feedback path by isolating heavy compute
 * (16x exp_pwl, 16x multiply) from memory port muxing.
 */
class AttnFullyFused : public match::Module {
  static const int kDebugLevel = 3;
  SC_HAS_PROCESS(AttnFullyFused);

public:
  // External ports (unchanged from single-thread version)
  Connections::In<spec::Axi::SubordinateToRVA::Write> rva_in;
  Connections::Out<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::In<bool> start;
  Connections::Out<bool> done;
  Connections::Out<spec::GB::Large::DataReq> large_req;
  Connections::In<spec::GB::Large::DataRsp<1>> large_rsp;

  // Internal channels between MemCtrl and Compute threads
  Connections::Combinational<NVUINT8> cfg_ch;
  Connections::Combinational<spec::VectorType> q_ch;
  Connections::Combinational<spec::VectorType> k_ch;
  Connections::Combinational<spec::VectorType> v_ch;
  Connections::Combinational<spec::VectorType> output_ch;

  // FSM enums at class scope for HLS compatibility
  enum MemFSM {
    M_IDLE, M_SEND_CFG, M_ROW_START,
    M_READ_Q_REQ, M_READ_Q_RSP, M_PUSH_Q,
    M_READ_K_REQ, M_READ_K_RSP, M_PUSH_K,
    M_READ_V_REQ, M_READ_V_RSP, M_PUSH_V,
    M_RECV_OUTPUT, M_WRITE_OUTPUT,
    M_NEXT_ROW, M_FIN
  };

  enum ComputeFSM {
    C_WAIT_CFG, C_WAIT_Q,
    C_RECV_K, C_DOT_SCALE,
    C_TILE_MAX, C_TILE_RESCALE, C_TILE_EXP, C_TILE_SUM,
    C_RECV_V, C_V_CONVERT, C_V_MAC,
    C_TILE_DONE,
    C_FINAL_RECIP, C_FINAL_NORM, C_SEND_OUTPUT
  };

  AttnFullyFused(sc_module_name nm) :
      match::Module(nm),
      rva_in("rva_in"),
      rva_out("rva_out"),
      start("start"),
      done("done"),
      large_req("large_req"),
      large_rsp("large_rsp"),
      cfg_ch("cfg_ch"),
      q_ch("q_ch"),
      k_ch("k_ch"),
      v_ch("v_ch"),
      output_ch("output_ch") {
    SC_THREAD(MemCtrlRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);

    SC_THREAD(ComputeRun);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  // =========================================================================
  // Thread 1: Memory Controller + AXI Config
  // Handles all GBCore reads/writes and external port communication.
  // Lightweight FSM — no heavy compute, small mux.
  // =========================================================================
  void MemCtrlRun() {
    // Reset external ports
    rva_in.Reset();
    rva_out.Reset();
    start.Reset();
    done.Reset();
    large_req.Reset();
    large_rsp.Reset();

    // Reset channel endpoints (this thread writes cfg/q/k/v, reads output)
    cfg_ch.ResetWrite();
    q_ch.ResetWrite();
    k_ch.ResetWrite();
    v_ch.ResetWrite();
    output_ch.ResetRead();

    MemFSM m_state = M_IDLE;
    spec::Attention::AttentionConfig attn_config;
    attn_config.Reset();

    bool is_start = 0;
    bool w_axi_rsp = 0;
    bool w_done = 0;
    spec::Axi::SubordinateToRVA::Read rva_out_reg;
    spec::GB::Large::DataReq large_req_reg;
    spec::VectorType mem_data_reg;

    NVUINT8 m_row_counter = 0;
    NVUINT8 m_col_counter = 0;
    NVUINT8 m_tile_start = 0;
    NVUINT8 m_tile_count = 0;
    NVUINT8 m_v_idx = 0;

#pragma hls_pipeline_init_interval 1
    while (1) {
      w_axi_rsp = 0;
      w_done = 0;

      // AXI config handling (priority over FSM)
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
        // Memory controller FSM
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

          case M_SEND_CFG: {
            cfg_ch.Push(attn_config.seq_len);
            m_state = M_ROW_START;
            break;
          }

          case M_ROW_START: {
            m_col_counter = 0;
            m_tile_start = 0;
            m_tile_count = 0;
            m_state = M_READ_Q_REQ;
            break;
          }

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
            if (large_rsp.PopNB(rsp)) {
              mem_data_reg = rsp.read_vector[0];
              m_state = M_PUSH_Q;
            }
            break;
          }

          case M_PUSH_Q: {
            q_ch.Push(mem_data_reg);
            m_state = M_READ_K_REQ;
            break;
          }

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
            if (large_rsp.PopNB(rsp)) {
              mem_data_reg = rsp.read_vector[0];
              m_state = M_PUSH_K;
            }
            break;
          }

          case M_PUSH_K: {
            k_ch.Push(mem_data_reg);
            m_col_counter++;
            m_tile_count++;
            bool tile_full = (m_tile_count >= spec::Attention::kTileSize);
            bool row_done = (m_col_counter >= attn_config.seq_len);
            if (tile_full || row_done) {
              m_v_idx = 0;
              m_state = M_READ_V_REQ;
            } else {
              m_state = M_READ_K_REQ;
            }
            break;
          }

          case M_READ_V_REQ: {
            NVUINT8 v_col = m_tile_start + m_v_idx;
            large_req_reg.is_write = 0;
            large_req_reg.memory_index = attn_config.v_memory_index;
            large_req_reg.vector_index = 0;
            large_req_reg.timestep_index = v_col;
            large_req.Push(large_req_reg);
            attn_config.gb_read_count++;
            m_state = M_READ_V_RSP;
            break;
          }

          case M_READ_V_RSP: {
            spec::GB::Large::DataRsp<1> rsp;
            if (large_rsp.PopNB(rsp)) {
              mem_data_reg = rsp.read_vector[0];
              m_state = M_PUSH_V;
            }
            break;
          }

          case M_PUSH_V: {
            v_ch.Push(mem_data_reg);
            m_v_idx++;
            if (m_v_idx >= m_tile_count) {
              // All V vectors for this tile sent
              m_tile_start = m_col_counter;
              m_tile_count = 0;
              if (m_col_counter >= attn_config.seq_len) {
                m_state = M_RECV_OUTPUT;
              } else {
                m_state = M_READ_K_REQ;
              }
            } else {
              m_state = M_READ_V_REQ;
            }
            break;
          }

          case M_RECV_OUTPUT: {
            spec::VectorType out_data;
            if (output_ch.PopNB(out_data)) {
              mem_data_reg = out_data;
              m_state = M_WRITE_OUTPUT;
            }
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
            m_state = M_NEXT_ROW;
            break;
          }

          case M_NEXT_ROW: {
            m_row_counter++;
            if (m_row_counter >= attn_config.seq_len) {
              m_state = M_FIN;
            } else {
              m_state = M_ROW_START;
            }
            break;
          }

          case M_FIN: {
            is_start = 0;
            w_done = 1;
            m_state = M_IDLE;
            break;
          }

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
  // Thread 2: Compute Datapath
  // Handles dot products, online softmax, V accumulation, normalization.
  // No GBCore access — all data arrives via internal channels.
  // Smaller mux (15 states, no memory port registers).
  // =========================================================================
  void ComputeRun() {
    // Reset channel endpoints (this thread reads cfg/q/k/v, writes output)
    cfg_ch.ResetRead();
    q_ch.ResetRead();
    k_ch.ResetRead();
    v_ch.ResetRead();
    output_ch.ResetWrite();

    ComputeFSM c_state = C_WAIT_CFG;

    NVUINT8 seq_len = 0;
    NVUINT8 row_counter = 0;
    NVUINT8 col_counter = 0;
    NVUINT8 tile_count = 0;
    NVUINT8 v_idx = 0;

    spec::VectorType q_row_reg;
    spec::VectorType k_data_reg;
    spec::VectorType v_data_reg;
    spec::VectorType write_data;

    spec::Attention::FixedType tile_logits[spec::Attention::kTileSize];
    spec::Attention::UnsignedFixedType tile_exp[spec::Attention::kTileSize];

    spec::Attention::FixedType running_max;
    spec::Attention::UnsignedAccumType running_sum;
    spec::Attention::UnsignedFixedType correction_reg;
    spec::Attention::AccumType sum_recip_reg;

    spec::Attention::AccumType v_accum[spec::kVectorSize];
    spec::Attention::FixedType v_fixed[spec::kVectorSize];

    // Initialize
    running_max = spec::kAttentionWordMin;
    running_sum = 0;
    correction_reg = 0;
#pragma hls_unroll yes
    for (int i = 0; i < spec::kVectorSize; i++) {
      v_accum[i] = 0;
      v_fixed[i] = 0;
    }
#pragma hls_unroll yes
    for (int i = 0; i < spec::Attention::kTileSize; i++) {
      tile_logits[i] = 0;
      tile_exp[i] = 0;
    }

#pragma hls_pipeline_init_interval 3
    while (1) {
      switch (c_state) {
        case C_WAIT_CFG: {
          NVUINT8 cfg;
          if (cfg_ch.PopNB(cfg)) {
            seq_len = cfg;
            row_counter = 0;
            c_state = C_WAIT_Q;
          }
          break;
        }

        case C_WAIT_Q: {
          spec::VectorType q_data;
          if (q_ch.PopNB(q_data)) {
            q_row_reg = q_data;
            col_counter = 0;
            tile_count = 0;
            running_max = spec::kAttentionWordMin;
            running_sum = 0;
#pragma hls_unroll yes
            for (int k = 0; k < spec::kVectorSize; k++) {
              v_accum[k] = 0;
            }
            c_state = C_RECV_K;
          }
          break;
        }

        case C_RECV_K: {
          spec::VectorType k_data;
          if (k_ch.PopNB(k_data)) {
            k_data_reg = k_data;
            c_state = C_DOT_SCALE;
          }
          break;
        }

        case C_DOT_SCALE: {
          spec::Attention::FixedType logit;
          AttnDotProductScale(q_row_reg, k_data_reg, logit);
          tile_logits[tile_count] = logit;
          tile_count++;
          col_counter++;
          bool tile_full = (tile_count >= spec::Attention::kTileSize);
          bool row_done = (col_counter >= seq_len);
          if (tile_full || row_done) {
            c_state = C_TILE_MAX;
          } else {
            c_state = C_RECV_K;
          }
          break;
        }

        case C_TILE_MAX: {
          spec::Attention::FixedType tile_max = spec::kAttentionWordMin;
#pragma hls_unroll yes
          for (int i = 0; i < spec::Attention::kTileSize; i++) {
            if (i < tile_count && tile_logits[i] > tile_max) {
              tile_max = tile_logits[i];
            }
          }
          spec::Attention::FixedType new_max = (tile_max > running_max) ? tile_max : running_max;
          spec::Attention::FixedType corr_input = running_max - new_max;
          correction_reg = ac_math::ac_exp_pwl<spec::Attention::UnsignedFixedType>(corr_input);
          running_max = new_max;
          c_state = C_TILE_RESCALE;
          break;
        }

        case C_TILE_RESCALE: {
#pragma hls_unroll yes
          for (int k = 0; k < spec::kVectorSize; k++) {
            v_accum[k] = v_accum[k] * correction_reg;
          }
          running_sum = running_sum * correction_reg;
          c_state = C_TILE_EXP;
          break;
        }

        case C_TILE_EXP: {
#pragma hls_unroll yes
          for (int i = 0; i < spec::Attention::kTileSize; i++) {
            if (i < tile_count) {
              spec::Attention::FixedType shifted = tile_logits[i] - running_max;
              tile_exp[i] = ac_math::ac_exp_pwl<spec::Attention::UnsignedFixedType>(shifted);
            } else {
              tile_exp[i] = 0;
            }
          }
          c_state = C_TILE_SUM;
          break;
        }

        case C_TILE_SUM: {
          // Use local variable with addtree to reduce chain depth on running_sum
          spec::Attention::UnsignedAccumType tile_sum = 0;
#pragma hls_unroll yes
#pragma cluster addtree
#pragma cluster_type both
          for (int i = 0; i < spec::Attention::kTileSize; i++) {
            tile_sum += tile_exp[i];
          }
          running_sum += tile_sum;
          v_idx = 0;
          c_state = C_RECV_V;
          break;
        }

        case C_RECV_V: {
          spec::VectorType v_data;
          if (v_ch.PopNB(v_data)) {
            v_data_reg = v_data;
            c_state = C_V_CONVERT;
          }
          break;
        }

        case C_V_CONVERT: {
          AttnConvertInputToFixed(v_data_reg, v_fixed);
          c_state = C_V_MAC;
          break;
        }

        case C_V_MAC: {
          AttnWeightedAccum(tile_exp[v_idx], v_fixed, v_accum);
          v_idx++;
          if (v_idx >= tile_count) {
            c_state = C_TILE_DONE;
          } else {
            c_state = C_RECV_V;
          }
          break;
        }

        case C_TILE_DONE: {
          tile_count = 0;
          if (col_counter >= seq_len) {
            c_state = C_FINAL_RECIP;
          } else {
            c_state = C_RECV_K;
          }
          break;
        }

        case C_FINAL_RECIP: {
          ac_math::ac_reciprocal_pwl(running_sum, sum_recip_reg);
          c_state = C_FINAL_NORM;
          break;
        }

        case C_FINAL_NORM: {
#pragma hls_unroll yes
          for (int k = 0; k < spec::kVectorSize; k++) {
            spec::Attention::FixedType normalized = v_accum[k] * sum_recip_reg;
            spec::NMP::InputFixedType out_tmp = ConvertToNmpOutputType(normalized);
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
            c_state = C_WAIT_Q;
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
