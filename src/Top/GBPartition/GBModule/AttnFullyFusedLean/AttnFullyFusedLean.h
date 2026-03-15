#ifndef __ATTN_FULLY_FUSED_LEAN__
#define __ATTN_FULLY_FUSED_LEAN__

#include <ac_math.h>
#include <nvhls_module.h>
#include <systemc.h>

#include "AttentionSpec.h"
#include "AttnDatapath.h"

/**
 * Fully Fused Attention — Single-Thread Lean Variant.
 *
 * Same algorithm as AttnFullyFused (online softmax + fused V accumulation)
 * but merged into a single SC_THREAD. Eliminates the Combinational channel
 * overhead between MemCtrl and Compute threads.
 *
 * Trade-off: May hit higher II (4 instead of 3) due to larger FSM mux
 * combining memory port logic with heavy compute (exp_pwl, rescaling).
 * But saves area from removed channel infrastructure.
 *
 * Data movement: Same as FullyFused — only reads Q,K,V and writes O.
 * Zero intermediate materialization. 9x write reduction vs Unfused.
 */
class AttnFullyFusedLean : public match::Module {
  static const int kDebugLevel = 3;
  SC_HAS_PROCESS(AttnFullyFusedLean);

public:
  Connections::In<spec::Axi::SubordinateToRVA::Write> rva_in;
  Connections::Out<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::In<bool> start;
  Connections::Out<bool> done;
  Connections::Out<spec::GB::Large::DataReq> large_req;
  Connections::In<spec::GB::Large::DataRsp<1>> large_rsp;

  enum FSM {
    IDLE,
    ROW_START,
    // Q read
    READ_Q_REQ, READ_Q_RSP,
    // K tile: read + dot product
    READ_K_REQ, READ_K_RSP,
    DOT_SCALE,
    // Online softmax per tile
    TILE_MAX,
    TILE_RESCALE,
    TILE_EXP,
    TILE_SUM,
    // V tile: read + MAC
    READ_V_REQ, READ_V_RSP,
    V_CONVERT,
    V_MAC,
    TILE_DONE,
    // Final output
    FINAL_RECIP,
    FINAL_NORM,
    WRITE_OUTPUT,
    NEXT_ROW,
    FIN
  };
  FSM state, next_state;

  bool is_start;
  spec::Attention::AttentionConfig attn_config;
  bool w_axi_rsp, w_done;
  spec::Axi::SubordinateToRVA::Read rva_out_reg;

  spec::GB::Large::DataReq large_req_reg;
  spec::GB::Large::DataRsp<1> large_rsp_reg;

  // Row state
  spec::VectorType q_row_reg;
  NVUINT8 col_counter;
  NVUINT8 tile_count;
  NVUINT8 tile_start;
  NVUINT8 v_idx;
  NVUINT8 rescale_idx;  // for serialized rescaling (0, 4, 8, 12)

  // Tile compute state
  spec::Attention::FixedType tile_logits[spec::Attention::kTileSize];
  spec::Attention::UnsignedFixedType tile_exp[spec::Attention::kTileSize];
  spec::Attention::FixedType running_max;
  spec::Attention::UnsignedAccumType running_sum;
  spec::Attention::UnsignedFixedType correction_reg;
  spec::Attention::AccumType sum_recip_reg;

  // V accumulation
  spec::Attention::AccumType v_accum[spec::kVectorSize];
  spec::Attention::FixedType v_fixed[spec::kVectorSize];
  spec::VectorType write_data;

  AttnFullyFusedLean(sc_module_name nm) :
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
    tile_start = 0;
    v_idx = 0;
    rescale_idx = 0;
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

  void RunFSM() {
    switch (state) {
      case IDLE: ResetCompute(); break;

      case ROW_START: {
        col_counter = 0;
        tile_count = 0;
        tile_start = 0;
        running_max = spec::kAttentionWordMin;
        running_sum = 0;
#pragma hls_unroll yes
        for (int k = 0; k < spec::kVectorSize; k++) v_accum[k] = 0;
        break;
      }

      // Read Q row
      case READ_Q_REQ: PushReadReq(attn_config.q_memory_index, 0, attn_config.row_counter); break;
      case READ_Q_RSP: break;

      // Read K vectors (one at a time within tile)
      case READ_K_REQ: PushReadReq(attn_config.k_memory_index, 0, col_counter); break;
      case READ_K_RSP: break;
      case DOT_SCALE: {
        spec::Attention::FixedType logit;
        AttnDotProductScale(q_row_reg, large_rsp_reg.read_vector[0], logit);
        tile_logits[tile_count] = logit;
        tile_count++;
        col_counter++;
        break;
      }

      // Online softmax for current tile
      case TILE_MAX: {
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
        break;
      }
      case TILE_RESCALE: {
        // Serialize rescaling: 4 elements per cycle instead of 16
        // Saves ~29% area (eliminates 12 of 16 large 40x40 multipliers)
        // Cost: 3 extra cycles per tile boundary (negligible for large N)
        NVUINT8 base = rescale_idx;
#pragma hls_unroll yes
        for (int k = 0; k < 4; k++) {
          v_accum[base + k] = v_accum[base + k] * correction_reg;
        }
        // Only rescale running_sum once (on last batch)
        if (rescale_idx == 0) {
          running_sum = running_sum * correction_reg;
        }
        rescale_idx += 4;
        break;
      }
      case TILE_EXP: {
#pragma hls_unroll yes
        for (int i = 0; i < spec::Attention::kTileSize; i++) {
          if (i < tile_count) {
            spec::Attention::FixedType shifted = tile_logits[i] - running_max;
            tile_exp[i] = ac_math::ac_exp_pwl<spec::Attention::UnsignedFixedType>(shifted);
          } else {
            tile_exp[i] = 0;
          }
        }
        break;
      }
      case TILE_SUM: {
        spec::Attention::UnsignedAccumType tile_sum = 0;
#pragma hls_unroll yes
#pragma cluster addtree
#pragma cluster_type both
        for (int i = 0; i < spec::Attention::kTileSize; i++) {
          tile_sum += tile_exp[i];
        }
        running_sum += tile_sum;
        v_idx = 0;
        break;
      }

      // Read V vectors for this tile and accumulate
      case READ_V_REQ: {
        NVUINT8 v_col = tile_start + v_idx;
        PushReadReq(attn_config.v_memory_index, 0, v_col);
        break;
      }
      case READ_V_RSP: break;
      case V_CONVERT: {
        AttnConvertInputToFixed(large_rsp_reg.read_vector[0], v_fixed);
        break;
      }
      case V_MAC: {
        AttnWeightedAccum(tile_exp[v_idx], v_fixed, v_accum);
        v_idx++;
        break;
      }
      case TILE_DONE: {
        tile_start = col_counter;
        tile_count = 0;
        break;
      }

      // Final normalization and output write
      case FINAL_RECIP: {
        ac_math::ac_reciprocal_pwl(running_sum, sum_recip_reg);
        break;
      }
      case FINAL_NORM: {
#pragma hls_unroll yes
        for (int k = 0; k < spec::kVectorSize; k++) {
          spec::Attention::FixedType normalized = v_accum[k] * sum_recip_reg;
          spec::NMP::InputFixedType out_tmp = ConvertToNmpOutputType(normalized);
          write_data[k] = nvhls::get_slc<spec::kIntWordWidth>(out_tmp, 0);
        }
        break;
      }
      case WRITE_OUTPUT: {
        PushWriteReq(attn_config.o_memory_index, 0, attn_config.row_counter, write_data);
        break;
      }
      case NEXT_ROW: break;

      case FIN: { is_start = 0; w_done = 1; break; }
      default: break;
    }
  }

  void UpdateFSM() {
    switch (state) {
      case IDLE: {
        bool start_reg;
        if (start.PopNB(start_reg)) is_start = attn_config.is_valid && start_reg;
        next_state = is_start ? ROW_START : IDLE;
        if (is_start) {
          attn_config.ResetCounter();
          attn_config.cycle_count = 0;
          attn_config.gb_read_count = 0;
          attn_config.gb_write_count = 0;
        }
        break;
      }

      case ROW_START: next_state = READ_Q_REQ; break;
      case READ_Q_REQ: next_state = READ_Q_RSP; break;
      case READ_Q_RSP: {
        spec::GB::Large::DataRsp<1> d;
        if (large_rsp.PopNB(d)) { q_row_reg = d.read_vector[0]; next_state = READ_K_REQ; }
        else next_state = READ_Q_RSP;
        break;
      }

      case READ_K_REQ: next_state = READ_K_RSP; break;
      case READ_K_RSP: {
        spec::GB::Large::DataRsp<1> d;
        if (large_rsp.PopNB(d)) { large_rsp_reg = d; next_state = DOT_SCALE; }
        else next_state = READ_K_RSP;
        break;
      }
      case DOT_SCALE: {
        bool tile_full = (tile_count >= spec::Attention::kTileSize);
        bool row_done = (col_counter >= attn_config.seq_len);
        if (tile_full || row_done) {
          next_state = TILE_MAX;
        } else {
          next_state = READ_K_REQ;
        }
        break;
      }

      case TILE_MAX: { rescale_idx = 0; next_state = TILE_RESCALE; break; }
      case TILE_RESCALE: {
        next_state = (rescale_idx >= spec::kVectorSize) ? TILE_EXP : TILE_RESCALE;
        break;
      }
      case TILE_EXP: next_state = TILE_SUM; break;
      case TILE_SUM: next_state = READ_V_REQ; break;

      case READ_V_REQ: next_state = READ_V_RSP; break;
      case READ_V_RSP: {
        spec::GB::Large::DataRsp<1> d;
        if (large_rsp.PopNB(d)) { large_rsp_reg = d; next_state = V_CONVERT; }
        else next_state = READ_V_RSP;
        break;
      }
      case V_CONVERT: next_state = V_MAC; break;
      case V_MAC: {
        if (v_idx >= tile_count) {
          next_state = TILE_DONE;
        } else {
          next_state = READ_V_REQ;
        }
        break;
      }
      case TILE_DONE: {
        if (col_counter >= attn_config.seq_len) {
          next_state = FINAL_RECIP;
        } else {
          next_state = READ_K_REQ;
        }
        break;
      }

      case FINAL_RECIP: next_state = FINAL_NORM; break;
      case FINAL_NORM: next_state = WRITE_OUTPUT; break;
      case WRITE_OUTPUT: next_state = NEXT_ROW; break;
      case NEXT_ROW: {
        bool row_end = 0;
        attn_config.UpdateRowCounter(row_end);
        next_state = row_end ? FIN : ROW_START;
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
