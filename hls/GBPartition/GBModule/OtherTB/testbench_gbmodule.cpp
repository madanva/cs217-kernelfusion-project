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

#include "GBModule.h"
#include <systemc.h>
#include <mc_scverify.h>
#include <testbench/nvhls_rand.h>
#include <nvhls_connections.h>
#include <map>
#include <vector>
#include <deque>
#include <utility>
#include <sstream>
#include <string>
#include <cstdlib>
#include <math.h> // testbench only
#include <queue>
#include <iomanip>
#include <nvhls_int.h>
#include <nvhls_types.h>
#include <nvhls_vector.h>
#include <cmath> // For std::sqrt

#include <iostream>

#include "SM6Spec.h"
#include "AxiSpec.h"
#include "AdpfloatSpec.h"
#include "AdpfloatUtils.h"
#include "GBSpec.h"


#define NVHLS_VERIFY_BLOCKS (GBModule)
#include <nvhls_verify.h>


#ifdef COV_ENABLE
   #pragma CTC SKIP
#endif

SC_MODULE(Source) {
 public:
  std::queue<spec::VectorType>* golden_queue;

  sc_in<bool> clk;
  sc_in<bool> rst;  
  Connections::Out<spec::StreamType> data_in;  
  Connections::Out<bool> pe_done; 
  Connections::Out<spec::Axi::SubordinateToRVA::Write> rva_in;

  SC_CTOR(Source) {
    SC_THREAD(layernorm_run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  void layernorm_run() {
    nvhls::set_random_seed();
    data_in.Reset();
    pe_done.Reset();
    rva_in.Reset();
    spec::Axi::SubordinateToRVA::Write rva_in_src;

    // --- Data Generation ---
    const int num_vectors = 16; // Number of vectors in the 16x16 matrix (rows)
    const int vector_size = 16; // Size of each vector (columns)
    std::vector<std::vector<double>> input_matrix(num_vectors, std::vector<double>(vector_size));
    std::vector<double> gamma_vector(vector_size);
    std::vector<double> beta_vector(vector_size);

    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < vector_size; ++j) {
            input_matrix[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0);
        }
    }
    for (int i = 0; i < vector_size; ++i) {
        gamma_vector[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0);
        beta_vector[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0);
    }

    AdpfloatType<8, 3> adpfloat_tmp;
    spec::AdpfloatBiasType adpbias_input = 2; // Bias for input
    spec::AdpfloatBiasType adpbias_gamma = 1; // Bias for gamma
    spec::AdpfloatBiasType adpbias_beta = 0;  // Bias for beta
    wait();

    // --- AXI Configuration ---

    // 1. Configure GBCore SmallBuffer base addresses for gamma and beta
    // This tells the GBCore where to find the start of gamma (manager 5) and beta (manager 6) data.
    // Address 0x400040 maps to GBCoreConfig, write_index=4 (which sets base_small array)
    rva_in_src.rw = 1;
    NVUINTW(128) base_small_config_data = 0;
    // Set base_small[5] = 0 (gamma starts at SmallBuffer base address)
    base_small_config_data.set_slc<16>(16 * 5, 0); 
    // Set base_small[6] = 16 (beta starts 16 vectors after gamma)
    base_small_config_data.set_slc<16>(16 * 6, 16); 
    rva_in_src.data = base_small_config_data;
    rva_in_src.addr = 0x400040; // GBCoreConfig, write_index=4 (sets base_small array)
    rva_in.Push(rva_in_src);
    wait();

    // 2. Configure LayerNorm module using GBControlConfig structure (address 0x900010)
    // This configures the LayerNorm operation within the GBModule.
    rva_in_src.rw = 1;
    NVUINTW(128) ln_gbcontrol_config_data = 0;
    ln_gbcontrol_config_data.set_slc<1>(0, 1);                            // is_valid = 1
    // mode=0 (Unidirectional from GBSpec.h), is_rnn=0, memory_index_1=0, memory_index_2=0
    ln_gbcontrol_config_data.set_slc<8>(48, num_vectors);                 // num_vector_1 = 16
    ln_gbcontrol_config_data.set_slc<16>(64, 1);                          // num_timestep_1 = 1
    ln_gbcontrol_config_data.set_slc<spec::kAdpfloatBiasWidth>(96, adpbias_input);  // adpbias_1 (input)
    ln_gbcontrol_config_data.set_slc<spec::kAdpfloatBiasWidth>(112, adpbias_beta);  // adpbias_3 (beta)
    ln_gbcontrol_config_data.set_slc<spec::kAdpfloatBiasWidth>(120, adpbias_gamma); // adpbias_4 (gamma)
    rva_in_src.data = ln_gbcontrol_config_data;
    rva_in_src.addr = 0x900010; // LayerNorm config (GBControlConfig for LayerNorm)
    rva_in.Push(rva_in_src);
    wait();

    // --- Data Loading ---
    spec::VectorType act_vec;

    // Load Input data to LargeBuffer (starting at address 0x500000 + 0)
    for (int i = 0; i < num_vectors; i++) {
        for (int k = 0; k < vector_size; k++) {
            adpfloat_tmp.Reset();
            adpfloat_tmp.set_value(input_matrix[i][k], adpbias_input);
            act_vec[k] = adpfloat_tmp.to_rawbits();
        }
        rva_in_src.rw = 1;
        rva_in_src.data = act_vec.to_rawbits();
        rva_in_src.addr = 0x500000 + i * 16; // Each vector takes 16 bytes (kNumVectorLanes * sizeof(ScalarType))
        rva_in.Push(rva_in_src);
        wait();
    }

    // Load Gamma vector to SmallBuffer (manager 5, starting at offset 0)
    for (int k = 0; k < vector_size; k++) {
        adpfloat_tmp.Reset();
        adpfloat_tmp.set_value(gamma_vector[k], adpbias_gamma);
        act_vec[k] = adpfloat_tmp.to_rawbits();
    }
    rva_in_src.rw = 1;
    rva_in_src.data = act_vec.to_rawbits();
    rva_in_src.addr = 0x600000 + 0; // SmallBuffer base + base_small[5] offset (0) * 16 bytes/vector
    rva_in.Push(rva_in_src);
    wait();

    // Load Beta vector to SmallBuffer (manager 6, starting at offset 16)
    for (int k = 0; k < vector_size; k++) {
        adpfloat_tmp.Reset();
        adpfloat_tmp.set_value(beta_vector[k], adpbias_beta);
        act_vec[k] = adpfloat_tmp.to_rawbits();
    }
    rva_in_src.rw = 1;
    rva_in_src.data = act_vec.to_rawbits();
    rva_in_src.addr = 0x600000 + 16 * 16; // SmallBuffer base + base_small[6] offset (16) * 16 bytes/vector
    rva_in.Push(rva_in_src);
    wait();

    // --- Golden Model ---
    const double epsilon = 1e-5; // Small constant for numerical stability
    for (int i = 0; i < num_vectors; ++i) {
        double sum = 0.0;
        for (int j = 0; j < vector_size; ++j) {
            sum += input_matrix[i][j];
        }
        double mean = sum / vector_size;

        double sq_sum = 0.0;
        for (int j = 0; j < vector_size; ++j) {
            sq_sum += (input_matrix[i][j] - mean) * (input_matrix[i][j] - mean);
        }
        double variance = sq_sum / vector_size;
        double std_dev = sqrt(variance + epsilon);

        spec::VectorType golden_vec;
        for (int j = 0; j < vector_size; ++j) {
            double normalized = (input_matrix[i][j] - mean) / std_dev;
            double golden_val = normalized * gamma_vector[j] + beta_vector[j];
            adpfloat_tmp.Reset();
            // Convert golden_val to AdpfloatType using the input bias
            adpfloat_tmp.set_value(golden_val, adpbias_input);
            golden_vec[j] = adpfloat_tmp.to_rawbits();
        }
        golden_queue->push(golden_vec);
    }
    
    // --- Start Signal ---
    // The LayerNorm module's FSM in LayerNorm.h indicates it starts when gbcontrol_config.is_valid is set.
    // The configuration write to 0x900010 already sets is_valid=1, so it should start automatically.
    // However, the testbench_layernorm.cpp sends a start signal to 0x000030. Let's send that too.
    rva_in_src.rw = 1;
    rva_in_src.data = 0; // Data doesn't matter for a start signal
    rva_in_src.addr = 0x000030; // Start signal for LayerNorm in GBModule
    rva_in.Push(rva_in_src);
    wait();
  }
};

SC_MODULE(Dest) {
 public:
  std::queue<spec::VectorType>* golden_queue;
  int error_count;

  sc_in<bool> clk;
  sc_in<bool> rst;
  Connections::In<spec::Axi::SubordinateToRVA::Read> rva_out;
  Connections::In<spec::StreamType> data_out; 
  Connections::In<bool> done; 
  Connections::In<bool> pe_start; 

  spec::Axi::SubordinateToRVA::Read rva_out_dest;
  spec::StreamType data_out_dest;
  bool done_dest;
  bool pe_start_dest;

  SC_CTOR(Dest) : error_count(0) {
    SC_THREAD(PopDataOut);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(PopDone);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(PopPEStart);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
    SC_THREAD(RVA_Out);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }

  ~Dest() {
    if (error_count > 0) {
      SC_REPORT_ERROR("testbench", "Mismatches found.");
    }
    if (!golden_queue->empty()) {
      SC_REPORT_ERROR("testbench", "Not all golden data was consumed.");
    }
  }   
  void RVA_Out() {
    rva_out.Reset();
    data_out.Reset();
    done.Reset();
    pe_start.Reset();
    
    wait();

    while (1) {
      if (rva_out.PopNB(rva_out_dest)) {
        // In this test, the output is expected to be written back to LargeBuffer, not streamed via rva_out.
        // This thread can be modified to read from the LargeBuffer if necessary.
      }
      wait(1);
    } 
  }

  void PopDataOut() {
   wait();
   while (1) {
     if (data_out.PopNB(data_out_dest)) {
        cout << sc_time_stamp() << " Design data_out result" << " \t " << endl;
        if (golden_queue->empty()) {
            SC_REPORT_ERROR("testbench", "Received unexpected data from DUT.");
            error_count++;
        } else {
            spec::VectorType golden_vec = golden_queue->front();
            golden_queue->pop();
            // The output of LayerNorm is written back to the LargeBuffer.
            // Data received on 'data_out' is likely unexpected or represents an intermediate stage.
            // For now, let's still compare, but this part of the testbench might need further refinement
            // to explicitly read the final output from the LargeBuffer.
            if (data_out_dest.data != golden_vec) {
                cout << "Mismatch!" << endl;
                cout << "DUT out:    " << data_out_dest.data << endl;
                cout << "Golden out: " << golden_vec << endl;
                error_count++;
            }
        }
        for (int i = 0; i < spec::kNumVectorLanes; i++) {
          AdpfloatType<8,3> tmp(data_out_dest.data[i]);
          cout << tmp.to_float(2) << endl; 
        }
     }
     wait(); 
   } 
  }

  void PopDone() {
   wait();
   while (1) {
     if (done.PopNB(done_dest)) {
        cout << sc_time_stamp() << " Done signal issued!!!" << " \t " << done_dest << endl;
     }
     wait(); 
   }
  }

  void PopPEStart() {
   wait();
   while (1) {
     if (pe_start.PopNB(pe_start_dest)) {
        cout << sc_time_stamp() << " PE_start signal issued!!!" << " \t " << pe_start_dest << endl;
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
  Connections::Combinational<spec::StreamType> data_in;     
  Connections::Combinational<spec::StreamType> data_out;     
  Connections::Combinational<bool> pe_start;  
  Connections::Combinational<bool> pe_done;  
  Connections::Combinational<bool> done;  

  NVHLS_DESIGN(GBModule) dut;
  Source  source;
  Dest    dest;

  std::queue<spec::VectorType> golden_queue;

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
    dut.rva_in(rva_in);
    dut.rva_out(rva_out);
    dut.data_out(data_out);
    dut.data_in(data_in);
    dut.done(done);
    dut.pe_done(pe_done);
    dut.pe_start(pe_start);

    source.clk(clk);
    source.rst(rst);
    source.data_in(data_in);
    source.pe_done(pe_done);
    source.rva_in(rva_in);
    source.golden_queue = &golden_queue;

    dest.clk(clk);
    dest.rst(rst);
    dest.rva_out(rva_out);
    dest.data_out(data_out);
    dest.pe_start(pe_start);
    dest.done(done);
    dest.golden_queue = &golden_queue;

    SC_THREAD(run); 
  }
  
  void run(){
    wait(2, SC_NS );
    std::cout << "@" << sc_time_stamp() <<" Asserting reset" << std::endl;
    rst.write(false);
    wait(2, SC_NS );
    rst.write(true);
    std::cout << "@" << sc_time_stamp() <<" De-Asserting reset" << std::endl;
    wait(100000, SC_NS );
    std::cout << "@" << sc_time_stamp() <<" sc_stop" << std::endl;
    sc_stop();
  }
};

int sc_main(int argc, char *argv[]) {
  
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
