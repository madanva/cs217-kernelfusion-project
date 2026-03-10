/*
 * GBModule Attention Integration Testbench
 *
 * Tests the full AXI-driven path through GBModule for attention:
 * 1. Write Q, K, V data into GBCore SRAM via AXI (region 0x5)
 * 2. Configure GBCore address mapping (region 0x4)
 * 3. Configure attention module (region 0xD)
 * 4. Trigger attention start (region 0x0, local_index 0x3)
 * 5. Wait for gb_done
 * 6. Read output from GBCore SRAM via AXI
 * 7. Compare against double-precision golden reference
 */

#include "GBModule.h"

#include <mc_scverify.h>
#include <nvhls_connections.h>
#include <systemc.h>
#include <testbench/nvhls_rand.h>

#include <cmath>
#include <deque>
#include <iostream>
#include <vector>

#include "AttentionSpec.h"
#include "GBSpec.h"
#include "NMPSpec.h"
#include "Spec.h"
#include "helper.h"

#define NVHLS_VERIFY_BLOCKS (GBModule)
#include <nvhls_verify.h>
#ifdef COV_ENABLE
#pragma CTC SKIP
#endif

// =============================================================================
// Test Parameters
// =============================================================================
static const int kSeqLen = 16;  // N=16 for integration test
static const int kHeadDim = 16;
static const int kNumTiles = 1;

// Tolerance
static const double kAbsTolerance = 0.5;
static const double kPctTolerance = 15.0;

// =============================================================================
// Global State
// =============================================================================
spec::VectorType Q_data[kSeqLen];
spec::VectorType K_data[kSeqLen];
spec::VectorType V_data[kSeqLen];
spec::VectorType expected_output[kSeqLen];

// Queues for Dest to compare
std::deque<spec::VectorType> expected_attn_outputs;
unsigned rva_read_count = 0;
bool attn_done_received = false;
int attn_output_mismatches = 0;
int attn_outputs_checked = 0;

// =============================================================================
// Helpers
// =============================================================================

NVINTW(spec::kIntWordWidth) float2fixed_attn(float val, int frac_bits) {
  return (NVINTW(spec::kIntWordWidth))(val * (1 << frac_bits));
}

bool vectors_match_with_tolerance(
    const spec::VectorType& actual,
    const spec::VectorType& expected) {
  bool ok = true;
  for (int i = 0; i < spec::kVectorSize; i++) {
    const double exp_val = fixed2float<spec::kIntWordWidth,
        spec::kIntWordWidth - spec::NMP::kNmpInputNumFrac>(expected[i]);
    const double act_val = fixed2float<spec::kIntWordWidth,
        spec::kIntWordWidth - spec::NMP::kNmpInputNumFrac>(actual[i]);
    const double abs_err = std::fabs(act_val - exp_val);
    const double denom = std::max(std::fabs(exp_val), 1e-9);
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
      Qf[i][j] = fixed2float<spec::kIntWordWidth,
          spec::kIntWordWidth - spec::NMP::kNmpInputNumFrac>(Q[i][j]);
      Kf[i][j] = fixed2float<spec::kIntWordWidth,
          spec::kIntWordWidth - spec::NMP::kNmpInputNumFrac>(K[i][j]);
      Vf[i][j] = fixed2float<spec::kIntWordWidth,
          spec::kIntWordWidth - spec::NMP::kNmpInputNumFrac>(V[i][j]);
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

inline NVUINTW(24) make_gbcore_data_addr(NVUINT16 local_index) {
  NVUINTW(24) addr = 0;
  addr.set_slc<4>(20, NVUINT4(0x5));
  addr.set_slc<16>(4, local_index);
  return addr;
}

NVUINTW(128) make_attn_cfg_data(
    uint8_t q_mem, uint8_t k_mem, uint8_t v_mem, uint8_t o_mem,
    uint8_t seq_len, uint8_t head_dim, uint8_t num_tiles) {
  NVUINTW(128) data = 0;
  data.set_slc<1>(0, NVUINT1(1));       // is_valid
  data.set_slc<3>(8, NVUINT3(q_mem));
  data.set_slc<3>(16, NVUINT3(k_mem));
  data.set_slc<3>(24, NVUINT3(v_mem));
  data.set_slc<3>(32, NVUINT3(o_mem));
  data.set_slc<8>(48, NVUINT8(seq_len));
  data.set_slc<8>(64, NVUINT8(head_dim));
  data.set_slc<8>(80, NVUINT8(num_tiles));
  return data;
}

// =============================================================================
// Source Module — drives AXI commands
// =============================================================================
SC_MODULE(Source) {
  sc_in<bool> clk;
  sc_in<bool> rst;
  Connections::Out<spec::Axi::SubordinateToRVA::Write> rva_in;
  Connections::Out<bool> pe_done;
  Connections::Out<spec::StreamType> data_in;

  SC_CTOR(Source) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void wait_for_read_response() {
    unsigned before = rva_read_count;
    while (rva_read_count == before) wait();
  }

  void run() {
    rva_in.Reset();
    pe_done.Reset();
    data_in.Reset();
    wait();

    spec::Axi::SubordinateToRVA::Write rva_cmd;

    // Generate random Q, K, V
    srand(42);
    for (int i = 0; i < kSeqLen; i++) {
      for (int j = 0; j < kHeadDim; j++) {
        Q_data[i][j] = (rand() % 8) - 4;
        K_data[i][j] = (rand() % 8) - 4;
        V_data[i][j] = (rand() % 8) - 4;
      }
    }
    compute_attention_golden(Q_data, K_data, V_data, expected_output,
                             kSeqLen, kHeadDim);
    cout << sc_time_stamp() << " Golden reference computed" << endl;

    // =========================================================================
    // Step 1: Configure GBCore address mapping (tag 0x4, local 0x1)
    // mem0(Q): base=0, num_vec=1
    // mem1(K): base=16, num_vec=1
    // mem2(V): base=32, num_vec=1
    // mem3(O): base=48, num_vec=1
    // =========================================================================
    NVUINTW(128) gbcore_addr_cfg = 0;
    // mem0
    gbcore_addr_cfg.set_slc<8>(0, NVUINT8(1));
    gbcore_addr_cfg.set_slc<16>(16, NVUINT16(0));
    // mem1
    gbcore_addr_cfg.set_slc<8>(32, NVUINT8(1));
    gbcore_addr_cfg.set_slc<16>(48, NVUINT16(16));
    // mem2
    gbcore_addr_cfg.set_slc<8>(64, NVUINT8(1));
    gbcore_addr_cfg.set_slc<16>(80, NVUINT16(32));
    // mem3
    gbcore_addr_cfg.set_slc<8>(96, NVUINT8(1));
    gbcore_addr_cfg.set_slc<16>(112, NVUINT16(48));

    rva_cmd.rw = 1;
    rva_cmd.data = gbcore_addr_cfg;
    rva_cmd.addr = set_bytes<3>("40_00_10");
    rva_in.Push(rva_cmd);
    cout << sc_time_stamp() << " GBCore address mapping configured" << endl;
    wait();

    // =========================================================================
    // Step 2: Write Q data to GBCore SRAM (physical addresses 0-15)
    // =========================================================================
    for (int i = 0; i < kSeqLen; i++) {
      rva_cmd.rw = 1;
      rva_cmd.data = Q_data[i].to_rawbits();
      rva_cmd.addr = make_gbcore_data_addr(i);
      rva_in.Push(rva_cmd);
      wait();
    }
    cout << sc_time_stamp() << " Q data written to GBCore" << endl;

    // =========================================================================
    // Step 3: Write K data (physical addresses 16-31)
    // =========================================================================
    for (int i = 0; i < kSeqLen; i++) {
      rva_cmd.rw = 1;
      rva_cmd.data = K_data[i].to_rawbits();
      rva_cmd.addr = make_gbcore_data_addr(16 + i);
      rva_in.Push(rva_cmd);
      wait();
    }
    cout << sc_time_stamp() << " K data written to GBCore" << endl;

    // =========================================================================
    // Step 4: Write V data (physical addresses 32-47)
    // =========================================================================
    for (int i = 0; i < kSeqLen; i++) {
      rva_cmd.rw = 1;
      rva_cmd.data = V_data[i].to_rawbits();
      rva_cmd.addr = make_gbcore_data_addr(32 + i);
      rva_in.Push(rva_cmd);
      wait();
    }
    cout << sc_time_stamp() << " V data written to GBCore" << endl;

    // =========================================================================
    // Step 5: Configure attention (tag 0xD, local 0x1)
    // =========================================================================
    rva_cmd.rw = 1;
    rva_cmd.data = make_attn_cfg_data(0, 1, 2, 3, kSeqLen, kHeadDim, kNumTiles);
    rva_cmd.addr = set_bytes<3>("D0_00_10");
    rva_in.Push(rva_cmd);
    cout << sc_time_stamp() << " Attention configured" << endl;
    wait();

    // =========================================================================
    // Step 6: Start attention (tag 0x0, local_index 0x3)
    // =========================================================================
    rva_cmd.rw = 1;
    rva_cmd.data = 0;
    rva_cmd.addr = set_bytes<3>("00_00_30");
    rva_in.Push(rva_cmd);
    cout << sc_time_stamp() << " Attention START triggered" << endl;

    // Wait for attention to complete
    while (!attn_done_received) wait();
    cout << sc_time_stamp() << " Attention DONE received, reading output" << endl;

    // Small delay to let any final writes settle
    wait(10);

    // =========================================================================
    // Step 7: Read output from GBCore (physical addresses 48-63)
    // =========================================================================
    for (int i = 0; i < kSeqLen; i++) {
      rva_cmd.rw = 0;
      rva_cmd.data = 0;
      rva_cmd.addr = make_gbcore_data_addr(48 + i);
      expected_attn_outputs.push_back(expected_output[i]);
      rva_in.Push(rva_cmd);
      wait_for_read_response();
    }
    cout << sc_time_stamp() << " All outputs read back" << endl;

    // =========================================================================
    // Step 8: Read performance counters (tag 0xD, local 0x2)
    // =========================================================================
    rva_cmd.rw = 0;
    rva_cmd.data = 0;
    rva_cmd.addr = set_bytes<3>("D0_00_20");
    rva_in.Push(rva_cmd);
    wait_for_read_response();

    wait(10);
    sc_stop();
  }
};

// =============================================================================
// Dest Module — receives AXI responses and checks results
// =============================================================================
SC_MODULE(Dest) {
  sc_in<bool> clk;
  sc_in<bool> rst;
  Connections::In<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::In<spec::StreamType> data_out;
  Connections::In<bool> pe_start;
  Connections::In<bool> gb_done;

  SC_CTOR(Dest) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);

    SC_THREAD(CheckDone);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);

    SC_THREAD(DrainUnused);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void CheckDone() {
    gb_done.Reset();
    pe_start.Reset();
    wait();

    bool reg;
    while (1) {
      if (gb_done.PopNB(reg)) {
        cout << sc_time_stamp() << " GB_DONE received!" << endl;
        attn_done_received = true;
      }
      if (pe_start.PopNB(reg)) {
        // drain pe_start if it fires
      }
      wait();
    }
  }

  void DrainUnused() {
    data_out.Reset();
    wait();

    spec::StreamType tmp;
    while (1) {
      data_out.PopNB(tmp);
      wait();
    }
  }

  void run() {
    rva_out.Reset();
    wait();

    while (1) {
      spec::Axi::SubordinateToRVA::Read rva_out_dest;
      if (rva_out.PopNB(rva_out_dest)) {
        rva_read_count++;

        if (!expected_attn_outputs.empty()) {
          // This is an attention output read — compare with tolerance
          spec::VectorType expected = expected_attn_outputs.front();
          expected_attn_outputs.pop_front();

          spec::VectorType actual;
          actual = rva_out_dest.data;

          attn_outputs_checked++;
          cout << sc_time_stamp() << " Checking output row "
               << attn_outputs_checked << "..." << endl;
          if (!vectors_match_with_tolerance(actual, expected)) {
            attn_output_mismatches++;
            SC_REPORT_ERROR("ATTN_INTEG", "Output mismatch");
          } else {
            cout << "  Row " << attn_outputs_checked << " PASSED" << endl;
          }
        } else {
          // Performance counter or other read
          NVUINT32 cycles = nvhls::get_slc<32>(rva_out_dest.data, 0);
          NVUINT32 reads  = nvhls::get_slc<32>(rva_out_dest.data, 32);
          NVUINT32 writes = nvhls::get_slc<32>(rva_out_dest.data, 64);
          if (cycles > 0 || reads > 0 || writes > 0) {
            cout << dec << sc_time_stamp() << " Performance counters:"
                 << " cycles=" << (int)cycles
                 << " gb_reads=" << (int)reads
                 << " gb_writes=" << (int)writes << endl;
          }
        }
      }
      wait();
    }
  }
};

// =============================================================================
// Testbench Top
// =============================================================================
SC_MODULE(testbench) {
  SC_HAS_PROCESS(testbench);

  sc_clock clk;
  sc_signal<bool> rst;

  Connections::Combinational<spec::Axi::SubordinateToRVA::Write> rva_in;
  Connections::Combinational<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::Combinational<spec::StreamType> data_in;
  Connections::Combinational<spec::StreamType> data_out;
  Connections::Combinational<bool> pe_start;
  Connections::Combinational<bool> pe_done;
  Connections::Combinational<bool> gb_done;

  NVHLS_DESIGN(GBModule) dut;
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
    dut.data_in(data_in);
    dut.data_out(data_out);
    dut.pe_start(pe_start);
    dut.pe_done(pe_done);
    dut.gb_done(gb_done);

    source.clk(clk);
    source.rst(rst);
    source.rva_in(rva_in);
    source.pe_done(pe_done);
    source.data_in(data_in);

    dest.clk(clk);
    dest.rst(rst);
    dest.rva_out(rva_out);
    dest.data_out(data_out);
    dest.pe_start(pe_start);
    dest.gb_done(gb_done);

    SC_THREAD(run);
  }

  void run() {
    wait(2, SC_NS);
    cout << "@" << sc_time_stamp() << " Asserting reset" << endl;
    rst.write(false);
    wait(2, SC_NS);
    rst.write(true);
    cout << "@" << sc_time_stamp() << " De-Asserting reset" << endl;
    // Generous timeout for N=16 through full GBModule
    wait(200000, SC_NS);
    cout << "@" << sc_time_stamp() << " TIMEOUT - sc_stop" << endl;
    sc_stop();
  }
};

// =============================================================================
// Entry Point
// =============================================================================
int sc_main(int argc, char* argv[]) {
  nvhls::set_random_seed();
  testbench tb("tb");
  sc_report_handler::set_actions(SC_ERROR, SC_DISPLAY);
  sc_start();

  bool rc = (sc_report_handler::get_count(SC_ERROR) > 0);
  cout << endl;
  cout << "========================================" << endl;
  cout << " Attention Integration Test: "
       << (rc ? "FAIL" : "PASS") << endl;
  cout << " Outputs checked: " << attn_outputs_checked << "/" << kSeqLen << endl;
  cout << " Mismatches: " << attn_output_mismatches << endl;
  cout << "========================================" << endl;
  if (rc)
    DCOUT("TESTBENCH FAIL" << endl);
  else
    DCOUT("TESTBENCH PASS" << endl);
  return rc;
}
