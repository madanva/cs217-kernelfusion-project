/*
 * Copyright 2026 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// =============================================================================
// GBModule Integration Testbench (GBCore + NMP)
// =============================================================================
//
// This testbench validates the integrated GBModule containing GBCore (SRAM
// scratchpad) and NMP (Near Memory Processing) submodules working together.
//
// Test Coverage:
// (a) AXI config read/write for GBCore and NMP configuration registers.
// (b) AXI direct read/write of GBCore large buffer SRAM.
// (c) Softmax operation via NMP with write-back to GBCore SRAM.
// (d) RMSNorm operation via NMP with write-back to GBCore SRAM.
// =============================================================================

#include "GBModule.h"

#include <mc_scverify.h>
#include <nvhls_connections.h>
#include <systemc.h>
#include <testbench/nvhls_rand.h>

#include <cmath>
#include <deque>
#include <iostream>
#include <vector>

#include <ac_math.h>
#include "AdpfloatSpec.h"
#include "AdpfloatUtils.h"
#include "GBSpec.h"
#include "NMPSpec.h"
#include "SM6Spec.h"
#include "helper.h"

#define NVHLS_VERIFY_BLOCKS (GBModule)
#include <nvhls_verify.h>

#ifdef COV_ENABLE
#pragma CTC SKIP
#endif

// =============================================================================
// Global State Variables
// =============================================================================

// Queue of expected AXI read responses (for config and direct SRAM reads)
std::deque<NVUINTW(128)> expected_rva_reads;
// Queue of expected NMP outputs (require tolerance-based comparison)
std::deque<spec::VectorType> expected_nmp_outputs;
// Corresponding adpfloat biases for NMP output decoding
std::deque<spec::AdpfloatBiasType> expected_nmp_biases;
// Counter for AXI read responses (used for Source/Dest synchronization)
unsigned rva_read_count = 0;

// =============================================================================
// Source Module
// =============================================================================

SC_MODULE(Source) {
  sc_in<bool> clk;
  sc_in<bool> rst;
  // AXI write interface for configuration and SRAM access
  Connections::Out<spec::Axi::SubordinateToRVA::Write> rva_in;
  // Start signal to trigger NMP operations
  Connections::Out<bool> start;
  // Done signal from NMP (operation complete)
  Connections::In<bool> done;

  SC_CTOR(Source) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  // ===========================================================================
  // Synchronization Helper Functions
  // ===========================================================================

  /**
   * @brief Block until NMP done signal is received.
   */
  void wait_for_done() {
    bool done_reg = 0;
    while (!done.PopNB(done_reg)) {
      wait();
    }
    while (done.PopNB(done_reg)) {
      wait();
    }
  }

  /**
   * @brief Drain any pending done signals.
   */
  void drain_done() {
    bool done_reg = 0;
    while (done.PopNB(done_reg)) {
      wait();
    }
  }

  /**
   * @brief Block until Dest module receives an AXI read response.
   */
  void wait_for_read_response() {
    unsigned before = rva_read_count;
    while (rva_read_count == before) {
      wait();
    }
  }

  // ===========================================================================
  // Main Test Sequence
  // ===========================================================================

  void run() {
    rva_in.Reset();
    start.Reset();
    done.Reset();
    wait();

    spec::Axi::SubordinateToRVA::Write rva_cmd;

    // (a) AXI config read/write for GBCore and NMP.
    cout << sc_time_stamp()
         << " Test (a): AXI config write/read for GBCore and NMP" << endl;
    NVUINTW(128) gbcore_cfg = make_gbcore_cfg_data(1, 0);
    rva_cmd.rw              = 1;
    rva_cmd.data            = gbcore_cfg;
    rva_cmd.addr            = set_bytes<3>("40_00_10");
    rva_in.Push(rva_cmd);
    wait();

    NVUINTW(128) nmp_cfg = make_nmp_cfg_data(1, 0, 1, 1, 0);
    rva_cmd.rw           = 1;
    rva_cmd.data         = nmp_cfg;
    rva_cmd.addr         = set_bytes<3>("C0_00_10");
    rva_in.Push(rva_cmd);
    wait();

    rva_cmd.rw   = 0;
    rva_cmd.data = 0;
    rva_cmd.addr = set_bytes<3>("40_00_10");
    rva_in.Push(rva_cmd);
    expected_rva_reads.push_back(gbcore_cfg);
    wait_for_read_response();

    rva_cmd.rw   = 0;
    rva_cmd.data = 0;
    rva_cmd.addr = set_bytes<3>("C0_00_10");
    rva_in.Push(rva_cmd);
    expected_rva_reads.push_back(nmp_cfg);
    wait_for_read_response();

    // (b) AXI read/write of GBCore large SRAM.
    cout << sc_time_stamp() << " Test (b): AXI write/read of GBCore large SRAM"
         << endl;
    for (NVUINT16 bank_idx = 0; bank_idx < spec::GB::Large::kNumBanks;
         bank_idx++) {
      spec::VectorType direct_data = 0;
      for (int i = 0; i < spec::kVectorSize; i++) {
        direct_data[i] = bank_idx + 1;
      }
      rva_cmd.rw   = 1;
      rva_cmd.data = direct_data.to_rawbits();
      rva_cmd.addr = make_gbcore_data_addr(bank_idx);
      rva_in.Push(rva_cmd);
      wait();

      rva_cmd.rw   = 0;
      rva_cmd.data = 0;
      rva_cmd.addr = make_gbcore_data_addr(bank_idx);
      rva_in.Push(rva_cmd);
      expected_rva_reads.push_back(direct_data.to_rawbits());
      wait_for_read_response();
    }

    // (c) Softmax operation on NMP and read back via GBCore.
    cout << sc_time_stamp() << " Test (c): NMP Softmax writeback to GBCore SRAM"
         << endl;
    std::vector<float> softmax_vals;
    for (int i = 0; i < spec::kVectorSize; i++) {
      softmax_vals.push_back((i - 7) * 0.12f);
    }
    spec::VectorType softmax_input = make_vector_from_floats(softmax_vals, 0);
    rva_cmd.rw                     = 1;
    rva_cmd.data                   = softmax_input.to_rawbits();
    rva_cmd.addr                   = set_bytes<3>("50_00_00");
    rva_in.Push(rva_cmd);
    wait();

    nmp_cfg      = make_nmp_cfg_data(1, 0, 1, 1, 0);
    rva_cmd.rw   = 1;
    rva_cmd.data = nmp_cfg;
    rva_cmd.addr = set_bytes<3>("C0_00_10");
    rva_in.Push(rva_cmd);
    wait();

    rva_cmd.rw   = 0;
    rva_cmd.data = 0;
    rva_cmd.addr = set_bytes<3>("C0_00_10");
    rva_in.Push(rva_cmd);
    expected_rva_reads.push_back(nmp_cfg);
    wait_for_read_response();

    drain_done();
    start.Push(1);
    wait(2);
    wait_for_done();

    spec::VectorType softmax_expected =
        compute_softmax_expected(softmax_vals, 0);
    rva_cmd.rw   = 0;
    rva_cmd.data = 0;
    rva_cmd.addr = set_bytes<3>("50_00_00");
    rva_in.Push(rva_cmd);
    // Use tolerance-based comparison for NMP outputs
    expected_nmp_outputs.push_back(softmax_expected);
    expected_nmp_biases.push_back(0);
    wait_for_read_response();

    // (d) RMSNorm operation on NMP and read back via GBCore.
    cout << sc_time_stamp() << " Test (d): NMP RMSNorm writeback to GBCore SRAM"
         << endl;
    std::vector<float> rms_vals;
    for (int i = 0; i < spec::kVectorSize; i++) {
      rms_vals.push_back((i + 1) * 0.03125f);
    }
    spec::VectorType rms_input = make_vector_from_floats(rms_vals, 0);
    rva_cmd.rw                 = 1;
    rva_cmd.data               = rms_input.to_rawbits();
    rva_cmd.addr               = set_bytes<3>("50_00_00");
    rva_in.Push(rva_cmd);
    wait();

    nmp_cfg      = make_nmp_cfg_data(0, 0, 1, 1, 0);
    rva_cmd.rw   = 1;
    rva_cmd.data = nmp_cfg;
    rva_cmd.addr = set_bytes<3>("C0_00_10");
    rva_in.Push(rva_cmd);
    wait();

    rva_cmd.rw   = 0;
    rva_cmd.data = 0;
    rva_cmd.addr = set_bytes<3>("C0_00_10");
    rva_in.Push(rva_cmd);
    expected_rva_reads.push_back(nmp_cfg);
    wait_for_read_response();

    drain_done();
    start.Push(1);
    wait(2);
    wait_for_done();

    spec::VectorType rms_expected = compute_rms_expected(rms_vals, 0);
    rva_cmd.rw                    = 0;
    rva_cmd.data                  = 0;
    rva_cmd.addr                  = set_bytes<3>("50_00_00");
    rva_in.Push(rva_cmd);
    // Use tolerance-based comparison for NMP outputs
    expected_nmp_outputs.push_back(rms_expected);
    expected_nmp_biases.push_back(0);
    wait_for_read_response();
  }
};

// =============================================================================
// Dest Module
// =============================================================================

SC_MODULE(Dest) {
  sc_in<bool> clk;
  sc_in<bool> rst;
  // AXI read response interface
  Connections::In<spec::Axi::SubordinateToRVA::Read> rva_out;

  SC_CTOR(Dest) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void run() {
    rva_out.Reset();
    rva_read_count = 0;
    wait();

    while (1) {
      spec::Axi::SubordinateToRVA::Read rva_out_dest;
      if (rva_out.PopNB(rva_out_dest)) {
        cout << hex << sc_time_stamp()
             << " RVA read data = " << rva_out_dest.data << endl;
        rva_read_count++;

        // Check if this is an NMP output requiring tolerance-based comparison
        if (!expected_nmp_outputs.empty()) {
          spec::VectorType expected   = expected_nmp_outputs.front();
          spec::AdpfloatBiasType bias = expected_nmp_biases.front();
          expected_nmp_outputs.pop_front();
          expected_nmp_biases.pop_front();

          spec::VectorType actual;
          actual = rva_out_dest.data;

          cout << sc_time_stamp() << " Comparing NMP output with tolerance..."
               << endl;
          if (!vectors_match_with_tolerance(actual, expected, bias)) {
            SC_REPORT_ERROR("GBModule", "NMP output mismatch");
          } else {
            cout << sc_time_stamp() << " NMP output matched within tolerance"
                 << endl;
          }
        } else if (!expected_rva_reads.empty()) {
          // Exact match for non-NMP reads (config, direct SRAM)
          NVUINTW(128) expected = expected_rva_reads.front();
          expected_rva_reads.pop_front();
          if (rva_out_dest.data != expected) {
            cout << hex << sc_time_stamp()
                 << " Expected RVA data = " << expected << endl;
            SC_REPORT_ERROR("GBModule", "RVA read mismatch");
          } else {
            cout << sc_time_stamp() << " RVA read matched" << endl;
          }
        } else {
          SC_REPORT_ERROR("GBModule", "Unexpected RVA read");
        }
      }
      wait();
    }
  }
};

// =============================================================================
// Testbench Top Module
// =============================================================================

SC_MODULE(testbench) {
  SC_HAS_PROCESS(testbench);

  // Clock and reset signals
  sc_clock clk;
  sc_signal<bool> rst;

  // AXI interface channels
  Connections::Combinational<spec::Axi::SubordinateToRVA::Write> rva_in;
  Connections::Combinational<spec::Axi::SubordinateToRVA::Read> rva_out;
  // NMP control signals
  Connections::Combinational<bool> start;
  Connections::Combinational<bool> done;

  // Module instances
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
    dut.start(start);
    dut.done(done);

    source.clk(clk);
    source.rst(rst);
    source.rva_in(rva_in);
    source.start(start);
    source.done(done);

    dest.clk(clk);
    dest.rst(rst);
    dest.rva_out(rva_out);

    SC_THREAD(run);
  }

  void run() {
    wait(2, SC_NS);
    std::cout << "@" << sc_time_stamp() << " Asserting reset" << std::endl;
    rst.write(false);
    wait(2, SC_NS);
    rst.write(true);
    std::cout << "@" << sc_time_stamp() << " De-Asserting reset" << std::endl;
    wait(1000, SC_NS);
    std::cout << "@" << sc_time_stamp() << " sc_stop" << std::endl;
    sc_stop();
  }
};

// =============================================================================
// Simulation Entry Point
// =============================================================================

int sc_main(int argc, char* argv[]) {
  // Initialize random seed for reproducible test patterns
  nvhls::set_random_seed();
  testbench tb("tb");

  // Configure error reporting to display but not abort
  sc_report_handler::set_actions(SC_ERROR, SC_DISPLAY);
  sc_start();

  // Return pass/fail based on error count
  bool rc = (sc_report_handler::get_count(SC_ERROR) > 0);
  if (rc)
    DCOUT("TESTBENCH FAIL" << endl);
  else
    DCOUT("TESTBENCH PASS" << endl);
  return rc;
}
