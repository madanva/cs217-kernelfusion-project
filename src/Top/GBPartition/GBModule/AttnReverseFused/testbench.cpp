/*
 * Attention Unit Testbench (Reverse Fused)
 *
 * Phase 1: QK^T (bulk writes of raw scores to mem3)
 * Phase 2: Fused Softmax + V multiply (reads scores back, writes output)
 *
 * Total writes to mem3: N*ceil(N/16) (QK^T scores) + N (output).
 * Only the last N writes (output) are checked against golden.
 */

#include <ac_math.h>
#include <cmath>
#include <mc_scverify.h>
#include <nvhls_connections.h>
#include <systemc.h>
#include <testbench/nvhls_rand.h>

#include <deque>
#include <sstream>
#include <string>
#include <vector>

#include "AxiSpec.h"
#include "GBSpec.h"
#include "AttentionSpec.h"
#include "NMPSpec.h"
#include "Spec.h"
#include "helper.h"
#include "AttnReverseFused.h"

#define NVHLS_VERIFY_BLOCKS (AttnReverseFused)
#include <nvhls_verify.h>
#ifdef COV_ENABLE
#pragma CTC SKIP
#endif

#ifndef ATTN_SEQ_LEN
#define ATTN_SEQ_LEN 64
#endif
static const int kSeqLen = ATTN_SEQ_LEN;
static const int kHeadDim = 16;

static const double kAbsTolerance = 0.5;
static const double kPctTolerance = 15.0;

spec::VectorType Q_data[kSeqLen];
spec::VectorType K_data[kSeqLen];
spec::VectorType V_data[kSeqLen];
spec::VectorType expected_output[kSeqLen];

NVUINTW(128) expected_cfg_data;
bool expected_cfg_valid = false;

bool seen_cfg_read = false;
bool seen_perf_read = false;
int output_rows_received = 0;
int total_writes = 0;

NVINTW(spec::kIntWordWidth) float2fixed_attn(float val, int frac_bits) {
  return (NVINTW(spec::kIntWordWidth))(val * (1 << frac_bits));
}

bool vectors_match_with_tolerance(
    const spec::VectorType& actual,
    const spec::VectorType& expected) {
  bool ok = true;
  for (int i = 0; i < spec::kVectorSize; i++) {
    const double exp_val = fixed2float<spec::kIntWordWidth, spec::kIntWordWidth - spec::NMP::kNmpInputNumFrac>(expected[i]);
    const double act_val = fixed2float<spec::kIntWordWidth, spec::kIntWordWidth - spec::NMP::kNmpInputNumFrac>(actual[i]);
    const double abs_err = std::fabs(act_val - exp_val);
    const double denom   = std::max(std::fabs(exp_val), 1e-9);
    const double pct_err = (abs_err / denom) * 100.0;
    const bool match = (abs_err <= kAbsTolerance) || (pct_err <= kPctTolerance);
    if (!match) {
      cout << "  Mismatch idx " << i
           << ": expected=" << exp_val << " actual=" << act_val
           << " abs_err=" << abs_err << " pct_err=" << pct_err << "%"
           << endl;
      ok = false;
    }
  }
  return ok;
}

void compute_attention_golden(
    spec::VectorType Q[], spec::VectorType K[], spec::VectorType V[],
    spec::VectorType out[], int N, int d) {

  std::vector<std::vector<double>> Qf(N, std::vector<double>(d));
  std::vector<std::vector<double>> Kf(N, std::vector<double>(d));
  std::vector<std::vector<double>> Vf(N, std::vector<double>(d));

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < d; j++) {
      Qf[i][j] = fixed2float<spec::kIntWordWidth, spec::kIntWordWidth - spec::NMP::kNmpInputNumFrac>(Q[i][j]);
      Kf[i][j] = fixed2float<spec::kIntWordWidth, spec::kIntWordWidth - spec::NMP::kNmpInputNumFrac>(K[i][j]);
      Vf[i][j] = fixed2float<spec::kIntWordWidth, spec::kIntWordWidth - spec::NMP::kNmpInputNumFrac>(V[i][j]);
    }
  }

  double inv_sqrt_d = 1.0 / std::sqrt((double)d);

  for (int i = 0; i < N; i++) {
    std::vector<double> logits(N);
    double max_val = -1e30;
    for (int j = 0; j < N; j++) {
      double dot = 0;
      for (int k = 0; k < d; k++) dot += Qf[i][k] * Kf[j][k];
      logits[j] = dot * inv_sqrt_d;
      if (logits[j] > max_val) max_val = logits[j];
    }

    double sum_exp = 0;
    std::vector<double> attn_weights(N);
    for (int j = 0; j < N; j++) {
      attn_weights[j] = std::exp(logits[j] - max_val);
      sum_exp += attn_weights[j];
    }
    for (int j = 0; j < N; j++) attn_weights[j] /= sum_exp;

    for (int k = 0; k < d; k++) {
      double val = 0;
      for (int j = 0; j < N; j++) val += attn_weights[j] * Vf[j][k];
      out[i][k] = float2fixed_attn(val, spec::NMP::kNmpInputNumFrac);
    }
  }
}

NVUINTW(128) make_attn_cfg_data(
    uint8_t q_mem, uint8_t k_mem, uint8_t v_mem, uint8_t o_mem,
    uint8_t seq_len, uint8_t head_dim, uint8_t num_tiles) {
  NVUINTW(128) data = 0;
  data.set_slc<1>(0, NVUINT1(1));
  data.set_slc<3>(8, NVUINT3(q_mem));
  data.set_slc<3>(16, NVUINT3(k_mem));
  data.set_slc<3>(24, NVUINT3(v_mem));
  data.set_slc<3>(32, NVUINT3(o_mem));
  data.set_slc<8>(48, NVUINT8(seq_len));
  data.set_slc<8>(64, NVUINT8(head_dim));
  data.set_slc<8>(80, NVUINT8(num_tiles));
  return data;
}

spec::Axi::SubordinateToRVA::Write make_attn_cfg(
    uint8_t q_mem, uint8_t k_mem, uint8_t v_mem, uint8_t o_mem,
    uint8_t seq_len, uint8_t head_dim, uint8_t num_tiles) {
  spec::Axi::SubordinateToRVA::Write w;
  w.rw = 1;
  w.data = make_attn_cfg_data(q_mem, k_mem, v_mem, o_mem, seq_len, head_dim, num_tiles);
  w.addr = set_bytes<3>("D0_00_10");
  return w;
}

spec::Axi::SubordinateToRVA::Write make_attn_cfg_read() {
  spec::Axi::SubordinateToRVA::Write w;
  w.rw = 0;
  w.addr = set_bytes<3>("D0_00_10");
  w.data = 0;
  return w;
}

spec::Axi::SubordinateToRVA::Write make_attn_perf_read() {
  spec::Axi::SubordinateToRVA::Write w;
  w.rw = 0;
  w.addr = set_bytes<3>("D0_00_20");
  w.data = 0;
  return w;
}

SC_MODULE(Source) {
  sc_in<bool> clk;
  sc_in<bool> rst;
  Connections::Out<spec::Axi::SubordinateToRVA::Write> rva_in;
  Connections::Out<bool> start;

  SC_CTOR(Source) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void run() {
    rva_in.Reset();
    start.Reset();
    wait();

    for (int i = 0; i < kSeqLen; i++) {
      for (int j = 0; j < kHeadDim; j++) {
        Q_data[i][j] = (rand() % 8) - 4;
        K_data[i][j] = (rand() % 8) - 4;
        V_data[i][j] = (rand() % 8) - 4;
      }
    }

    compute_attention_golden(Q_data, K_data, V_data, expected_output, kSeqLen, kHeadDim);

    spec::Axi::SubordinateToRVA::Write cfg = make_attn_cfg(
        0, 1, 2, 3, kSeqLen, kHeadDim, (kSeqLen + 15) / 16);
    expected_cfg_data = make_attn_cfg_data(
        0, 1, 2, 3, kSeqLen, kHeadDim, (kSeqLen + 15) / 16);
    expected_cfg_valid = true;
    rva_in.Push(cfg);
    wait(2);

    rva_in.Push(make_attn_cfg_read());
    wait(20);

    start.Push(true);
    wait(2);

    wait(400000);
    rva_in.Push(make_attn_perf_read());
    wait();
  }
};

SC_MODULE(Dest) {
  sc_in<bool> clk;
  sc_in<bool> rst;
  Connections::In<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::In<bool> done;
  Connections::In<spec::GB::Large::DataReq> large_req;
  Connections::Out<spec::GB::Large::DataRsp<1>> large_rsp;

  // GBCore storage for QK^T scores (N timesteps x ceil(N/16) vectors)
  spec::VectorType gbcore_storage[spec::Attention::kMaxSeqLen][spec::Attention::kMaxSeqLen / spec::kVectorSize];

  SC_CTOR(Dest) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void run() {
    rva_out.Reset();
    done.Reset();
    large_req.Reset();
    large_rsp.Reset();
    wait();

    // Reverse Fused writes:
    // Phase 1: N * ceil(N/16) score vectors to mem3
    // Phase 2: N output vectors to mem3
    // Total = N*ceil(N/16) + N
    static const int kNumVecsPerRow = (kSeqLen + spec::kVectorSize - 1) / spec::kVectorSize;
    static const int kPhase1Writes = kSeqLen * kNumVecsPerRow;

    while (1) {
      spec::GB::Large::DataReq req;
      spec::Axi::SubordinateToRVA::Read rva_out_dest;
      bool done_dest;

      if (large_req.PopNB(req)) {
        if (req.is_write) {
          NVUINT3 mem_idx = req.memory_index;
          NVUINT8 ts_idx = req.timestep_index;
          NVUINT8 vec_idx = req.vector_index;

          if (mem_idx == 3) {
            gbcore_storage[ts_idx][vec_idx] = req.write_data;
            total_writes++;

            cout << sc_time_stamp() << " Write #" << total_writes
                 << " to mem3 ts=" << (int)ts_idx
                 << " vec=" << (int)vec_idx << endl;

            // Check output writes (after Phase 1 score writes)
            if (total_writes > kPhase1Writes) {
              NVUINT8 row = ts_idx;
              cout << sc_time_stamp() << " Checking output row " << (int)row << endl;

              if (!vectors_match_with_tolerance(req.write_data, expected_output[row])) {
                SC_REPORT_ERROR("ATTN", "Output data mismatch");
              } else {
                cout << "  Row " << (int)row << " PASSED" << endl;
              }
              output_rows_received++;
            }
          }
        } else {
          // Read request
          NVUINT3 mem_idx = req.memory_index;
          NVUINT16 ts_idx = req.timestep_index;
          NVUINT8 vec_idx = req.vector_index;
          spec::GB::Large::DataRsp<1> rsp;

          if (mem_idx == 0) {
            rsp.read_vector[0] = Q_data[ts_idx];
          } else if (mem_idx == 1) {
            rsp.read_vector[0] = K_data[ts_idx];
          } else if (mem_idx == 2) {
            rsp.read_vector[0] = V_data[ts_idx];
          } else if (mem_idx == 3) {
            rsp.read_vector[0] = gbcore_storage[ts_idx][vec_idx];
          } else {
            rsp.read_vector[0] = 0;
          }
          large_rsp.Push(rsp);
        }
      }

      if (rva_out.PopNB(rva_out_dest)) {
        cout << hex << sc_time_stamp()
             << " AXI read response data = " << rva_out_dest.data << endl;

        if (expected_cfg_valid && !seen_cfg_read) {
          if (rva_out_dest.data != expected_cfg_data) {
            SC_REPORT_ERROR("ATTN", "Config readback mismatch");
          } else {
            cout << sc_time_stamp() << " Config readback matched" << endl;
          }
          seen_cfg_read = true;
        } else if (!seen_perf_read) {
          NVUINT32 cycles = nvhls::get_slc<32>(rva_out_dest.data, 0);
          NVUINT32 reads  = nvhls::get_slc<32>(rva_out_dest.data, 32);
          NVUINT32 writes = nvhls::get_slc<32>(rva_out_dest.data, 64);
          cout << dec << sc_time_stamp() << " Performance counters:"
               << " cycles=" << (int)cycles
               << " gb_reads=" << (int)reads
               << " gb_writes=" << (int)writes << endl;
          seen_perf_read = true;
        }
      }

      if (done.PopNB(done_dest)) {
        cout << dec << sc_time_stamp() << " Attention DONE signal received!" << endl;
        cout << "Total output rows checked: " << output_rows_received << endl;
        cout << "Total writes to mem3: " << total_writes << endl;
      }

      wait();
    }
  }
};

SC_MODULE(testbench) {
  SC_HAS_PROCESS(testbench);

  sc_clock clk;
  sc_signal<bool> rst;

  Connections::Combinational<spec::Axi::SubordinateToRVA::Write> rva_in;
  Connections::Combinational<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::Combinational<bool> start;
  Connections::Combinational<bool> done;
  Connections::Combinational<spec::GB::Large::DataReq> large_req;
  Connections::Combinational<spec::GB::Large::DataRsp<1>> large_rsp;

  NVHLS_DESIGN(AttnReverseFused) dut;
  Source source;
  Dest dest;

  testbench(sc_module_name name) :
      sc_module(name),
      clk("clk", 1.0, SC_NS, 0.5, 0, SC_NS, true),
      rst("rst"),
      dut("dut"),
      source("source"),
      dest("dest") {
    dut.clk(clk);
    dut.rst(rst);
    dut.rva_in(rva_in);
    dut.rva_out(rva_out);
    dut.start(start);
    dut.done(done);
    dut.large_req(large_req);
    dut.large_rsp(large_rsp);

    source.clk(clk);
    source.rst(rst);
    source.rva_in(rva_in);
    source.start(start);

    dest.clk(clk);
    dest.rst(rst);
    dest.rva_out(rva_out);
    dest.done(done);
    dest.large_req(large_req);
    dest.large_rsp(large_rsp);

    SC_THREAD(run);
  }

  void run() {
    wait(2, SC_NS);
    cout << "@" << sc_time_stamp() << " Asserting reset" << endl;
    rst.write(false);
    wait(2, SC_NS);
    rst.write(true);
    cout << "@" << sc_time_stamp() << " De-Asserting reset" << endl;
    wait(500000, SC_NS);
    cout << "@" << sc_time_stamp() << " sc_stop" << endl;
    sc_stop();
  }
};

int sc_main(int argc, char* argv[]) {
  nvhls::set_random_seed();
  testbench tb("tb");
  sc_report_handler::set_actions(SC_ERROR, SC_DISPLAY);
  sc_start();

  bool rc = (sc_report_handler::get_count(SC_ERROR) > 0);
  if (rc)
    DCOUT("TESTBENCH FAIL" << endl);
  else
    DCOUT("TESTBENCH PASS" << endl);
  return rc;
}
