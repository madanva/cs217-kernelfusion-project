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
// NMP Unit Testbench (AXI + RMSNorm + Softmax)
// =============================================================================
// This testbench validates the NMP module by exercising:
// - AXI config write/readback for NMP configuration registers.
// - RMSNorm processing on a randomized input vector.
// - Softmax processing on a deterministic, numerically stable vector.
// =============================================================================

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

#include "AdpfloatSpec.h"
#include "AdpfloatUtils.h"
#include "AxiSpec.h"
#include "GBSpec.h"
#include "NMP.h"
#include "NMPSpec.h"
#include "SM6Spec.h"
#include "helper.h"

#define NVHLS_VERIFY_BLOCKS (NMP)
#include <nvhls_verify.h>
#ifdef COV_ENABLE
#pragma CTC SKIP
#endif

// =============================================================================
// NMP Helper Functions
// =============================================================================

/**
 * @brief Create AXI write command with NMP configuration.
 * @return Complete AXI write request struct
 */
spec::Axi::SubordinateToRVA::Write make_cfg(
    uint8_t mode,
    uint8_t mem,
    uint8_t nvec,
    uint16_t ntimestep,
    uint8_t adpbias) {
  spec::Axi::SubordinateToRVA::Write w;
  w.rw   = 1;
  w.data = make_nmp_cfg_data(mode, mem, nvec, ntimestep, adpbias);
  w.addr = set_bytes<3>("C0_00_10");
  return w;
}

/**
 * @brief Create AXI read command for NMP configuration register.
 * @return AXI read request struct
 */
spec::Axi::SubordinateToRVA::Write make_cfg_read() {
  spec::Axi::SubordinateToRVA::Write w;
  w.rw   = 0;
  w.addr = set_bytes<3>("C0_00_10");
  w.data = 0;
  return w;
}

// =============================================================================
// Data Conversion Helper Functions
// =============================================================================

/**
 * @brief Convert float vector to adpfloat-encoded DataRsp for NMP input.
 * @param vals Input float values
 * @param bias Adpfloat bias for encoding
 * @return DataRsp struct with encoded vector
 */
spec::GB::Large::DataRsp<1> make_vec(
    const std::vector<float>& vals, spec::AdpfloatBiasType bias) {
  spec::GB::Large::DataRsp<1> rsp;
  rsp.read_vector[0] = make_vector_from_floats(vals, bias);
  return rsp;
}

// =============================================================================
// Global State Variables
// =============================================================================

// Expected AXI config data for readback verification
NVUINTW(128) expected_cfg_data;
// Flag indicating config data has been written
bool expected_cfg_valid = false;
// Expected RMSNorm output (golden reference)
spec::VectorType expected_rms_data;
// Expected Softmax output (golden reference)
spec::VectorType expected_softmax_data;
// Flags for tracking test progress
bool expected_rms_valid     = false;
bool expected_softmax_valid = false;
bool seen_cfg_read          = false;
bool seen_rms_write         = false;
bool seen_softmax_write     = false;

// =============================================================================
// Source Module
// =============================================================================

SC_MODULE(Source) {
  // Clock and reset signals
  sc_in<bool> clk;
  sc_in<bool> rst;
  // AXI write interface for configuration
  Connections::Out<spec::Axi::SubordinateToRVA::Write> rva_in;
  // Start signal to trigger NMP operation
  Connections::Out<bool> start;
  // Input data interface (simulates GBCore read response)
  Connections::Out<spec::GB::Large::DataRsp<1>> large_rsp;

  // Internal state for stimulus data
  std::vector<spec::Axi::SubordinateToRVA::Write> src_vec;
  bool start_src;
  spec::GB::Large::DataRsp<1> large_rsp_src;

  SC_CTOR(Source) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void run() {
    rva_in.Reset();
    start.Reset();
    large_rsp.Reset();
    wait();

    // Test 1: AXI RW test - write config
    spec::Axi::SubordinateToRVA::Write rva_in_src;
    rva_in_src         = make_cfg(0, 3, 2, 4, 5);
    expected_cfg_data  = make_nmp_cfg_data(0, 3, 2, 4, 5);
    expected_cfg_valid = true;
    rva_in.Push(rva_in_src);
    wait(2);

    // AXI RW test - read config back
    rva_in.Push(make_cfg_read());
    wait(20);

    // Test 2: RMSNorm test
    std::vector<float> rms_vals;
    for (int i = 0; i < spec::kVectorSize; i++) {
      int rand_val = nvhls::get_rand<8>().to_int();
      rms_vals.push_back((rand_val % 32) * 0.03125f);
    }
    expected_rms_data  = compute_rms_expected(rms_vals, 0);
    expected_rms_valid = true;
    rva_in_src         = make_cfg(0, 1, 1, 1, 0);
    rva_in.Push(rva_in_src);
    wait();

    start_src = 1;
    start.Push(start_src);
    wait(4);

    large_rsp_src = make_vec(rms_vals, 0);
    large_rsp.Push(large_rsp_src);
    wait(50);

    // Test 3: Softmax test
    std::vector<float> softmax_vals;
    for (int i = 0; i < spec::kVectorSize; i++) {
      softmax_vals.push_back((i - 7) * 0.12f);
    }
    expected_softmax_data  = compute_softmax_expected(softmax_vals, 0);
    expected_softmax_valid = true;
    rva_in_src             = make_cfg(1, 2, 1, 1, 0);
    rva_in.Push(rva_in_src);
    wait();

    start_src = 1;
    start.Push(start_src);
    wait(4);

    large_rsp_src = make_vec(softmax_vals, 0);
    large_rsp.Push(large_rsp_src);
    wait();
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
  // Done signal from NMP (operation complete)
  Connections::In<bool> done;
  // Write-back request interface (NMP output data)
  Connections::In<spec::GB::Large::DataReq> large_req;
  // Storage for received responses
  std::vector<spec::Axi::SubordinateToRVA::Read> dest_vec;

  SC_CTOR(Dest) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void run() {
    rva_out.Reset();
    done.Reset();
    large_req.Reset();
    wait();

    while (1) {
      spec::Axi::SubordinateToRVA::Read rva_out_dest;
      spec::GB::Large::DataReq large_req_dest;
      bool done_dest;

      if (large_req.PopNB(large_req_dest)) {
        cout << sc_time_stamp() << " - large buffer request sent: "
             << " - is_write: " << large_req_dest.is_write
             << " memory_index: " << large_req_dest.memory_index
             << " vector_index: " << large_req_dest.vector_index
             << " timestep_index: " << large_req_dest.timestep_index << endl;
        if (large_req_dest.is_write) {
          if (expected_rms_valid && !seen_rms_write) {
            if (!vectors_match_with_tolerance(
                    large_req_dest.write_data, expected_rms_data, 0)) {
              SC_REPORT_ERROR("NMP", "RMS write data mismatch");
            } else {
              cout << sc_time_stamp() << " RMS write data matched" << endl;
            }
            seen_rms_write = true;
          } else if (expected_softmax_valid && !seen_softmax_write) {
            if (!vectors_match_with_tolerance(
                    large_req_dest.write_data, expected_softmax_data, 0)) {
              SC_REPORT_ERROR("NMP", "Softmax write data mismatch");
            } else {
              cout << sc_time_stamp() << " Softmax write data matched" << endl;
            }
            seen_softmax_write = true;
          }
        }
      }
      if (rva_out.PopNB(rva_out_dest)) {
        cout << hex << sc_time_stamp()
             << " Dest rva data = " << rva_out_dest.data << endl;
        if (expected_cfg_valid && !seen_cfg_read) {
          if (rva_out_dest.data != expected_cfg_data) {
            SC_REPORT_ERROR("NMP", "RVA config readback mismatch");
          } else {
            cout << sc_time_stamp() << " RVA config matched" << endl;
          }
          seen_cfg_read = true;
        }
      }

      if (done.PopNB(done_dest)) {
        cout << hex << sc_time_stamp() << " Done signal issued !!!!" << endl;
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
  // Control signals
  Connections::Combinational<bool> start;
  Connections::Combinational<bool> done;
  // GBCore interface (simulated by testbench)
  Connections::Combinational<spec::GB::Large::DataReq> large_req;
  Connections::Combinational<spec::GB::Large::DataRsp<1>> large_rsp;

  // Module instances
  NVHLS_DESIGN(NMP) dut;
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
    source.large_rsp(large_rsp);

    dest.clk(clk);
    dest.rst(rst);
    dest.rva_out(rva_out);
    dest.done(done);
    dest.large_req(large_req);

    SC_THREAD(run);
  }

  void run() {
    wait(2, SC_NS);
    std::cout << "@" << sc_time_stamp() << " Asserting reset" << std::endl;
    rst.write(false);
    wait(2, SC_NS);
    rst.write(true);
    std::cout << "@" << sc_time_stamp() << " De-Asserting reset" << std::endl;
    wait(10000, SC_NS);
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
