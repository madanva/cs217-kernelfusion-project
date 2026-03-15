#ifndef __ATTN_REVERSE_FUSED__
#define __ATTN_REVERSE_FUSED__

#include <ac_math.h>
#include <nvhls_module.h>
#include <systemc.h>

#include "AttentionSpec.h"
#include "AttnDatapath.h"

/**
 * Reverse Fused Attention: QK^T separate, Softmax+V fused.
 *
 * Phase 1 (QK^T — bulk, like Unfused):
 *   For each Q row i, read all K rows, compute dot products, write raw
 *   score row as int8 vectors to scratchpad. Bulk reads then bulk writes
 *   per row — no port contention.
 *
 * Phase 2 (Softmax + V — fused):
 *   For each row i, read score vectors from scratchpad, compute online
 *   softmax (tile-by-tile with running max/sum), then read V rows and
 *   accumulate weighted sum. Write only the final output row.
 *
 * Key advantage: No write/read interleaving on GBCore port — eliminates
 * the PopNB stall that cripples PartialFused.
 *
 * Data movement: writes = N*ceil(N/16) (QK^T scores) + N (output).
 * Same write count as PartialFused, but no stall penalty.
 */
class AttnReverseFused : public match::Module {
  static const int kDebugLevel = 3;
  SC_HAS_PROCESS(AttnReverseFused);

public:
  Connections::In<spec::Axi::SubordinateToRVA::Write> rva_in;
  Connections::Out<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::In<bool> start;
  Connections::Out<bool> done;
  Connections::Out<spec::GB::Large::DataReq> large_req;
  Connections::In<spec::GB::Large::DataRsp<1>> large_rsp;

  enum FSM {
    IDLE,
    // Phase 1: QK^T (bulk reads, bulk writes per row)
    QKT_ROW_START,
    QKT_READ_Q_REQ, QKT_READ_Q_RSP,
    QKT_READ_K_REQ, QKT_READ_K_RSP,
    QKT_DOT_SCALE,
    QKT_PACK_VEC,
    QKT_WRITE_VEC,
    QKT_NEXT_ROW,
    // Phase 2: Fused Softmax + V multiply
    SV_ROW_START,
    SV_READ_SCORE_REQ, SV_READ_SCORE_RSP,
    SV_UNPACK_SCORE,
    SV_TILE_MAX,
    SV_TILE_RESCALE,
    SV_TILE_EXP,
    SV_TILE_SUM,
    SV_READ_V_REQ, SV_READ_V_RSP,
    SV_V_MAC,
    SV_TILE_DONE,
    SV_FINAL_RECIP,
    SV_FINAL_NORM,
    SV_WRITE_OUTPUT,
    SV_NEXT_ROW,
    FIN
  };
  FSM state, next_state;

  bool is_start;
  spec::Attention::AttentionConfig attn_config;
  bool w_axi_rsp, w_done;
  spec::Axi::SubordinateToRVA::Read rva_out_reg;

  spec::GB::Large::DataReq large_req_reg;
  spec::GB::Large::DataRsp<1> large_rsp_reg;

  // Phase 1 state
  spec::VectorType q_row_reg;
  spec::Attention::FixedType score_buf[spec::Attention::kMaxSeqLen];
  NVUINT8 col_counter;
  spec::VectorType write_data;
  NVUINT8 vec_idx;

  // Phase 2 state: online softmax + V accumulation (like FullyFused compute)
  spec::Attention::FixedType tile_logits[spec::Attention::kTileSize];
  spec::Attention::UnsignedFixedType tile_exp[spec::Attention::kTileSize];
  spec::Attention::FixedType running_max;
  spec::Attention::UnsignedAccumType running_sum;
  spec::Attention::UnsignedFixedType correction_reg;
  spec::Attention::AccumType sum_recip_reg;
  spec::Attention::AccumType v_accum[spec::kVectorSize];
  spec::Attention::FixedType v_fixed[spec::kVectorSize];
  NVUINT8 tile_count;
  NVUINT8 tile_start;
  NVUINT8 v_idx;
  NVUINT8 score_vec_idx;  // for reading score vectors in Phase 2

  AttnReverseFused(sc_module_name nm) :
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
    tile_count = 0;
    tile_start = 0;
    v_idx = 0;
    score_vec_idx = 0;
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

      // =====================================================================
      // Phase 1: QK^T (identical to Unfused Stage 1)
      // Bulk reads (Q row + all K rows), bulk writes (score vectors)
      // =====================================================================
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

      // =====================================================================
      // Phase 2: Fused Softmax + V multiply
      // Read score vectors → online softmax (tile-by-tile) → read V → accumulate
      // All sequential reads, no write/read interleaving
      // =====================================================================
      case SV_ROW_START: {
        score_vec_idx = 0;
        tile_count = 0;
        tile_start = 0;
        col_counter = 0;
        running_max = spec::kAttentionWordMin;
        running_sum = 0;
#pragma hls_unroll yes
        for (int k = 0; k < spec::kVectorSize; k++) v_accum[k] = 0;
        break;
      }

      // Read score vectors from scratchpad and unpack into tile_logits
      case SV_READ_SCORE_REQ: {
        PushReadReq(attn_config.o_memory_index, score_vec_idx, attn_config.row_counter);
        break;
      }
      case SV_READ_SCORE_RSP: break;
      case SV_UNPACK_SCORE: {
        // Unpack int8 scores into fixed-point tile_logits
        NVUINT8 base = score_vec_idx * spec::kVectorSize;
#pragma hls_unroll yes
        for (int k = 0; k < spec::kVectorSize; k++) {
          if ((base + k) < attn_config.seq_len) {
            spec::ScalarType tmp = large_rsp_reg.read_vector[0][k];
            NVINTW(spec::kIntWordWidth) signed_input = (NVINTW(spec::kIntWordWidth))tmp;
            spec::Attention::InputFixedType in_fixed = 0;
            in_fixed.set_slc(0, signed_input);
            tile_logits[k] = ConvertFromNmpInputType(in_fixed);
          }
        }
        // Count how many valid elements in this tile
        NVUINT8 remaining = attn_config.seq_len - (score_vec_idx * spec::kVectorSize);
        tile_count = (remaining >= spec::kVectorSize) ? (NVUINT8)spec::kVectorSize : remaining;
        tile_start = score_vec_idx * spec::kVectorSize;
        score_vec_idx++;
        break;
      }

      // Online softmax: tile max → rescale → exp → sum (same as FullyFused)
      case SV_TILE_MAX: {
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
      case SV_TILE_RESCALE: {
#pragma hls_unroll yes
        for (int k = 0; k < spec::kVectorSize; k++) {
          v_accum[k] = v_accum[k] * correction_reg;
        }
        running_sum = running_sum * correction_reg;
        break;
      }
      case SV_TILE_EXP: {
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
      case SV_TILE_SUM: {
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
      case SV_READ_V_REQ: {
        NVUINT8 v_col = tile_start + v_idx;
        PushReadReq(attn_config.v_memory_index, 0, v_col);
        break;
      }
      case SV_READ_V_RSP: break;
      case SV_V_MAC: {
        AttnConvertInputToFixed(large_rsp_reg.read_vector[0], v_fixed);
        AttnWeightedAccum(tile_exp[v_idx], v_fixed, v_accum);
        v_idx++;
        break;
      }
      case SV_TILE_DONE: {
        tile_count = 0;
        break;
      }

      // Final normalization and output
      case SV_FINAL_RECIP: {
        ac_math::ac_reciprocal_pwl(running_sum, sum_recip_reg);
        break;
      }
      case SV_FINAL_NORM: {
#pragma hls_unroll yes
        for (int k = 0; k < spec::kVectorSize; k++) {
          spec::Attention::FixedType normalized = v_accum[k] * sum_recip_reg;
          spec::NMP::InputFixedType out_tmp = ConvertToNmpOutputType(normalized);
          write_data[k] = nvhls::get_slc<spec::kIntWordWidth>(out_tmp, 0);
        }
        break;
      }
      case SV_WRITE_OUTPUT: {
        PushWriteReq(attn_config.o_memory_index, 0, attn_config.row_counter, write_data);
        break;
      }
      case SV_NEXT_ROW: break;

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

      // Phase 1: QK^T (same transitions as Unfused Stage 1)
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
        // After all QK^T rows, switch to Phase 2
        next_state = row_end ? SV_ROW_START : QKT_ROW_START;
        break;
      }

      // Phase 2: Fused Softmax + V multiply
      case SV_ROW_START: next_state = SV_READ_SCORE_REQ; break;
      case SV_READ_SCORE_REQ: next_state = SV_READ_SCORE_RSP; break;
      case SV_READ_SCORE_RSP: {
        spec::GB::Large::DataRsp<1> d;
        if (large_rsp.PopNB(d)) { large_rsp_reg = d; next_state = SV_UNPACK_SCORE; }
        else next_state = SV_READ_SCORE_RSP;
        break;
      }
      case SV_UNPACK_SCORE: {
        // After unpacking this tile's scores, proceed to softmax
        next_state = SV_TILE_MAX;
        break;
      }
      case SV_TILE_MAX: next_state = SV_TILE_RESCALE; break;
      case SV_TILE_RESCALE: next_state = SV_TILE_EXP; break;
      case SV_TILE_EXP: next_state = SV_TILE_SUM; break;
      case SV_TILE_SUM: next_state = SV_READ_V_REQ; break;
      case SV_READ_V_REQ: next_state = SV_READ_V_RSP; break;
      case SV_READ_V_RSP: {
        spec::GB::Large::DataRsp<1> d;
        if (large_rsp.PopNB(d)) { large_rsp_reg = d; next_state = SV_V_MAC; }
        else next_state = SV_READ_V_RSP;
        break;
      }
      case SV_V_MAC: {
        if (v_idx >= tile_count) {
          next_state = SV_TILE_DONE;
        } else {
          next_state = SV_READ_V_REQ;
        }
        break;
      }
      case SV_TILE_DONE: {
        // More score tiles to process?
        if (score_vec_idx < num_vecs) {
          next_state = SV_READ_SCORE_REQ;
        } else {
          next_state = SV_FINAL_RECIP;
        }
        break;
      }
      case SV_FINAL_RECIP: next_state = SV_FINAL_NORM; break;
      case SV_FINAL_NORM: next_state = SV_WRITE_OUTPUT; break;
      case SV_WRITE_OUTPUT: next_state = SV_NEXT_ROW; break;
      case SV_NEXT_ROW: {
        bool row_end = 0;
        attn_config.UpdateRowCounter(row_end);
        next_state = row_end ? FIN : SV_ROW_START;
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
