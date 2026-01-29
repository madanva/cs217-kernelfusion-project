/*
 * All rights reserved - Stanford University. 
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the "License"); 
 * you may not use this file except in compliance with the License.  
 * You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "PEModule.h"
#include <systemc.h>
#include <mc_scverify.h>
#include <testbench/nvhls_rand.h>
#include <nvhls_connections.h>
#include <vector>
#include <string>
#include <cstdlib>
#include <math.h> // testbench only
#include <iomanip>
#include <iostream>
#include <fstream>

#include "SM6Spec.h"
#include "AxiSpec.h"
#include "AdpfloatSpec.h"
#include "AdpfloatUtils.h"
#include "helper.h"

#define NVHLS_VERIFY_BLOCKS (PEModule)
#include <nvhls_verify.h>

std::vector<double> golden_y;
std::vector<NVUINTW(spec::VectorType::width)> golden_rva_out;

SC_MODULE(Source) {
  sc_in<bool> clk;
  sc_in<bool> rst;
  Connections::Out<spec::StreamType> input_port;
  Connections::Out<bool> start;
  Connections::Out<spec::Axi::SubordinateToRVA::Write> rva_in;

  SC_CTOR(Source) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void run() {
    start.Reset();
    input_port.Reset();
    rva_in.Reset();
    wait();

    const int size = 16;
    spec::AdpfloatBiasType adpbias = 2;
    
    // Generate random data
    std::vector<std::vector<double>> W(size, std::vector<double>(size));
    std::vector<double> x(size);

    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        W[i][j] = (double)rand() / RAND_MAX - 0.5;
      }
      x[i] = (double)rand() / RAND_MAX - 0.5;
    }

    // Golden model: y = tanh(W*x)
    std::vector<double> Wx = MatrixVectorMul(W, x);
    golden_y = VectorTanh(Wx);

    spec::Axi::SubordinateToRVA::Write rva_in_src;

    // 1. PEConfig: is_valid=1, is_bias=0, num_manager=1, num_output=1
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("00_00_00_00_00_00_00_00_00_00_01_01_00_00_00_01");
    rva_in_src.addr = set_bytes<3>("40_00_10");
    rva_in.Push(rva_in_src);
    /*cout << sc_time_stamp() << " Wrote PEConfig = " << rva_in_src.data << endl;
    wait();
    // Read
    rva_in_src.rw = 0;
    rva_in_src.addr = set_bytes<3>("40_00_10");
    rva_in.Push(rva_in_src);
    golden_rva_out.push_back(rva_in_src.data);*/
    wait();

    // 2. PEManager 0: num_input=16, base_weight=0, base_bias=0, base_input=0
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("00_00_00_00_00_00_00_00_10_00_00_00_02_02_02_00"); // num_input = 16 (0x10)
    rva_in_src.addr = set_bytes<3>("40_00_20");
    rva_in.Push(rva_in_src);
    cout << sc_time_stamp() << " Wrote PEManager 0 = " << rva_in_src.data << endl;
    wait();
    // Read
    rva_in_src.rw = 0;
    rva_in_src.addr = set_bytes<3>("40_00_20");
    rva_in.Push(rva_in_src);
    golden_rva_out.push_back(rva_in_src.data);
    wait();

    // 3. Load Weights (W)
    AdpfloatType<8, 3> adpfloat_tmp;
    for (int i = 0; i < size; i++) { // For each row of W
      spec::VectorType weight_vec;
      for (int j = 0; j < size; j++) {
        adpfloat_tmp.set_value(W[i][j], adpbias);
        weight_vec[j] = adpfloat_tmp.to_rawbits();
      }
      rva_in_src.rw = 1;
      rva_in_src.data = weight_vec.to_rawbits();
      rva_in_src.addr = 0x500000 + (i * 16); // base_weight=0, addr=0+i
      rva_in.Push(rva_in_src);
      wait();
    }
    
    // 4. Load Input (x)
    spec::VectorType input_vec;
    for (int i = 0; i < size; i++) {
        adpfloat_tmp.set_value(x[i], adpbias);
        input_vec[i] = adpfloat_tmp.to_rawbits();
    }
    rva_in_src.rw = 1;
    rva_in_src.data = input_vec.to_rawbits();
    rva_in_src.addr = 0x600000 + 0; // base_input=0
    rva_in.Push(rva_in_src);
    wait();

    // 5. ActUnit Config: num_inst=3, num_output=1
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("00_00_00_00_00_00_00_00_00_00_00_01_03_02_00_01");
    rva_in_src.addr = set_bytes<3>("80_00_10");
    rva_in.Push(rva_in_src);
    wait();

    // 6. ActUnit Instructions: INPE, TANH, OUTGB
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("00_00_00_00_00_00_00_00_00_00_00_00_00_40_B0_30");
    rva_in_src.addr = set_bytes<3>("80_00_20");
    rva_in.Push(rva_in_src);
    wait();

    // 7. Start
    rva_in_src.rw = 1;
    rva_in_src.data = 0;
    rva_in_src.addr = set_bytes<3>("00_00_00");
    rva_in.Push(rva_in_src);
    wait();

  }
};

SC_MODULE(Dest) {
  sc_in<bool> clk;
  sc_in<bool> rst;
  Connections::In<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::In<spec::StreamType> output_port;
  Connections::In<bool> done;

  std::vector<double> dut_output;

  SC_CTOR(Dest) {
    SC_THREAD(PopOutport);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(PopDone);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(PopRvaOut);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void PopRvaOut() {
    rva_out.Reset();
    wait();

    while (1) {
      spec::Axi::SubordinateToRVA::Read rva_out_dest;
      unsigned int i = 0;
      if (rva_out.PopNB(rva_out_dest)) {

        if (rva_out_dest.data != golden_rva_out[i]) {
            cout << "ERROR: MISMATCH in rva_out data. DUT=" << rva_out_dest.data 
                 << ", Golden=" << golden_rva_out[i] << endl;
            sc_assert(false);
        } else {
            cout << "rva_out data matches golden model: " << rva_out_dest.data << endl;
        }
        i++;
      }
      wait();
    }
  }

  void PopOutport() {
    output_port.Reset();
    wait();

    while (1) {
      spec::StreamType output_port_dest;
      if (output_port.PopNB(output_port_dest)) {
        for (int i = 0; i < spec::kNumVectorLanes; i++) {
          AdpfloatType<8,3> tmp(output_port_dest.data[i]);
          dut_output.push_back(tmp.to_float(2));
        }
      }
      wait();
    }
  }

  void PopDone() {
    done.Reset();
    wait();
    while (1) {
      bool done_dest;
      if (done.PopNB(done_dest)) {
        if (done_dest) {
            cout << "Starting comparison with golden model" << endl;
            bool passed = true;
            if (dut_output.size() != golden_y.size()) {
                cout << "ERROR: DUT output size (" << dut_output.size() << ") != golden model size (" << golden_y.size() << ")" << endl;
                passed = false;
            } else {
                double max_diff = 0.0;
                for (size_t i = 0; i < dut_output.size(); ++i) {
                    double diff = std::abs(dut_output[i] - golden_y[i]);
                    if (diff > max_diff) max_diff = diff;
                    if (diff > 1e-1) { // Looser tolerance for basic test
                        cout << "MISMATCH at index " << i << ": DUT=" << dut_output[i] << ", Golden=" << golden_y[i] << endl;
                        passed = false;
                    }
                }
                cout << "Max difference: " << max_diff << endl;
            }

            if (passed) {
                cout << "TESTBENCH PASSED" << endl;
            } else {
                cout << "TESTBENCH FAILED" << endl;
                sc_assert(false);
            }
            sc_stop();
        }

        
      }
      wait();
    }
  }
};

SC_MODULE(testbench) {
  SC_HAS_PROCESS(testbench);
  sc_clock clk;
  sc_signal<bool> rst;

  Connections::Combinational<spec::StreamType> input_port;
  Connections::Combinational<bool> start;
  Connections::Combinational<spec::Axi::SubordinateToRVA::Write> rva_in;
  Connections::Combinational<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::Combinational<spec::StreamType> output_port;
  Connections::Combinational<bool> done;

  NVHLS_DESIGN(PEModule) dut;
  Source  source;
  Dest    dest;

  testbench(sc_module_name name)
  : sc_module(name),
    clk("clk", 1.0, SC_NS, 0.5, 0, SC_NS, true),
    rst("rst"),
    dut("dut"),
    source("source"),
    dest("dest")
   {
     dut.clk(clk);
     dut.rst(rst);
     dut.input_port(input_port);
     dut.start(start);
     dut.rva_in(rva_in);
     dut.rva_out(rva_out);
     dut.output_port(output_port);
     dut.done(done);

     source.clk(clk);
     source.rst(rst);
     source.input_port(input_port);
     source.start(start);
     source.rva_in(rva_in);

     dest.clk(clk);
     dest.rst(rst);
     dest.rva_out(rva_out);
     dest.output_port(output_port);
     dest.done(done);

     SC_THREAD(run);
   }

  void run(){
    wait(2, SC_NS );
    std::cout << "@" << sc_time_stamp() <<" Asserting reset" << std::endl;
    rst.write(false);
    wait(10, SC_NS );
    rst.write(true);
    std::cout << "@" << sc_time_stamp() <<" De-Asserting reset" << std::endl;
    wait(5000, SC_NS );
    cout << "Error: Simulation timed out!" << endl;
    sc_stop();
  }
};

int sc_main(int argc, char *argv[]) {
  nvhls::set_random_seed();
  testbench tb("tb");
  sc_report_handler::set_actions(SC_ERROR, SC_DISPLAY);
  sc_start();
  bool rc = (sc_report_handler::get_count(SC_ERROR) > 0);
  if (rc)
    DCOUT("TESTBENCH FAIL" << endl);
  return rc;
}