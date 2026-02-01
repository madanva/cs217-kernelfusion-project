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

#include "SM6Spec.h"
#include "AxiSpec.h"
#include "AdpfloatSpec.h"
#include "AdpfloatUtils.h"

#include "helper.h"

#define NVHLS_VERIFY_BLOCKS (GBModule)
#include <nvhls_verify.h>


#ifdef COV_ENABLE
   #pragma CTC SKIP
#endif

std::vector<double> golden_y;

SC_MODULE(Source) {
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
    data_in.Reset();
    pe_done.Reset();
    rva_in.Reset();
    spec::Axi::SubordinateToRVA::Write  rva_in_src;

    spec::VectorType act_vec;
    spec::VectorType gamma_vec;
    spec::VectorType beta_vec;

    AdpfloatType<spec::kAdpfloatWordWidth, spec::kAdpfloatExpWidth> adpfloat_tmp;
    spec::AdpfloatBiasType adpbias_input = 2;
    spec::AdpfloatBiasType adpbias_beta = 0;
    spec::AdpfloatBiasType adpbias_gamma = 1;

    const int num_vectors = 16; // Number of vectors in the 16x16 matrix (rows)
    const int vector_size = 16; // Size of each vector (columns)
    std::vector<std::vector<double>> input_matrix(num_vectors, std::vector<double>(vector_size));
    std::vector<double> gamma_vector(vector_size);
    std::vector<double> beta_vector(vector_size);

    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < vector_size; ++j) {
            input_matrix[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.95; 
        }
    }
    for (int i = 0; i < vector_size; ++i) {
        gamma_vector[i] = 1;
        beta_vector[i] = 0;
    }

    // --- Golden Model (Fixed-Point Accurate) ---
    golden_y.clear();

    std::vector<spec::ActVectorType> fixed_input_matrix(num_vectors);
    spec::ActVectorType fixed_gamma_vector;
    spec::ActVectorType fixed_beta_vector;
    AdpfloatType<spec::kAdpfloatWordWidth, spec::kAdpfloatExpWidth> adp_converter;

    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < vector_size; ++j) {
            adp_converter.set_value(input_matrix[i][j], adpbias_input);
            fixed_input_matrix[i][j] = adp_converter.to_fixed<spec::kActWordWidth, spec::kActNumFrac>(adpbias_input);
        }
    }
    for (int i = 0; i < vector_size; ++i) {
        adp_converter.set_value(gamma_vector[i], adpbias_gamma);
        fixed_gamma_vector[i] = adp_converter.to_fixed<spec::kActWordWidth, spec::kActNumFrac>(adpbias_gamma);
        adp_converter.set_value(beta_vector[i], adpbias_beta);
        fixed_beta_vector[i] = adp_converter.to_fixed<spec::kActWordWidth, spec::kActNumFrac>(adpbias_beta);
    }

    spec::LayerNormSumType total_sum = 0;
    spec::LayerNormSumType total_sqsum = 0;

    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < vector_size; ++j) {
            total_sum += fixed_input_matrix[i][j];
            spec::ActScalarType val = fixed_input_matrix[i][j];
            NVINTW(spec::kActWordWidth * 2) sq_val_tmp = val * val;
            sq_val_tmp >>= spec::kActNumFrac;
            spec::ActScalarType sq_val = sq_val_tmp;
            total_sqsum += sq_val;
        }
    }

    NVUINT14 divisor = num_vectors * vector_size;
    spec::ActScalarType mean = total_sum / divisor;
    spec::ActScalarType sqmean = total_sqsum / divisor;

    NVINTW(2 * spec::kActWordWidth) meansq_tmp = mean * mean;
    meansq_tmp >>= spec::kActNumFrac;
    spec::ActScalarType meansq = meansq_tmp;

    spec::ActScalarType var = sqmean - meansq;

    double var_float = (double)var.to_int64() / (double)(1 << spec::kActNumFrac);
    double inv_std_float = 1.0 / sqrt(var_float + 1e-5);
    spec::ActScalarType inv_std = (spec::ActScalarType)(inv_std_float * (1 << spec::kActNumFrac));

    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < vector_size; ++j) {
            spec::ActScalarType normalized_val = fixed_input_matrix[i][j] - mean;
            
            NVINTW(spec::kActWordWidth * 2) mul_tmp = normalized_val * inv_std;
            mul_tmp >>= spec::kActNumFrac;
            normalized_val = mul_tmp;

            mul_tmp = normalized_val * fixed_gamma_vector[j];
            mul_tmp >>= spec::kActNumFrac;
            
            spec::ActScalarType scaled_val = mul_tmp;
            spec::ActScalarType final_val = scaled_val + fixed_beta_vector[j];

            AdpfloatType<spec::kAdpfloatWordWidth, spec::kAdpfloatExpWidth> final_adp;
            final_adp.set_value_fixed<spec::kActWordWidth, spec::kActNumFrac>(final_val, adpbias_input);
            
            golden_y.push_back(final_adp.to_float(adpbias_input));
        }
    }

    wait();

    //LayerNorm AXI GBControlConfig
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("01_00_00_02_00_00_00_10_00_01_00_00_00_00_00_01"); //is_valid=1, mode=0, is_rnn=0, memory_index=0, output_memory_index=0, num_vector= 16, num_output_vector=0, num_timestep=256, num_timestep_padding=0, adpbias_act=2, adpbias_atten=0, adpbias_beta = 0, adpbias_gamma=1
    rva_in_src.addr = set_bytes<3>("90_00_10");  // last 4 bits never used 
    rva_in.Push(rva_in_src);
    wait(); 

    //GBCore LargeBuffer Config
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_10"); //num_vector_large for kMaxNumManagers=0 is 16, base_large for kMaxNumManagers=0 is 0 
    rva_in_src.addr = set_bytes<3>("40_00_10");  // last 4 bits never used 
    rva_in.Push(rva_in_src);
    wait(); 

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
    wait(); 

    //GBCore SmallBuffer Config
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("00_00_00_10_00_00_00_00_00_00_00_00_00_00_00_00"); //base_small[5] for gamma = 0, base_small[6] for beta = 16 
    rva_in_src.addr = set_bytes<3>("40_00_20");  // GBCoreConfig, write_index=4 
    rva_in.Push(rva_in_src);
    wait(); 

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

    //send LayerNorm start signal
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00"); 
    rva_in_src.addr = set_bytes<3>("00_00_30");  
    rva_in.Push(rva_in_src);

    wait();

    //GBControl AXI GBControlConfig 
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("01_00_00_02_00_00_00_10_00_01_00_00_00_00_00_01"); //is_valid=1, mode=0, sendback=0, memory_index=0, output_memory_index=0, num_vector= 16, num_output_vector=0, num_timestep=320, num_timestep_padding=0, adpbias_act=2, adpbias_atten=0, adpbias_beta = 0, adpbias_gamma=1
    rva_in_src.addr = set_bytes<3>("70_00_10");  // last 4 bits never used 
    rva_in.Push(rva_in_src);
    wait(); 

    //send GBControl start signal
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00"); 
    rva_in_src.addr = set_bytes<3>("00_00_10");  
    rva_in.Push(rva_in_src);
    wait(); 
 
  } //layernorm_run

}; //SC MODULE Source

SC_MODULE(Dest) {
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

  bool done_PopDataOut = false;
  bool done_PopDone = false;
  bool done_PopPEStart = false;

  std::vector<double> Output;
  static int golden_y_offset;

  SC_CTOR(Dest) {
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
    SC_THREAD(SimStop);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);

  }   
  void RVA_Out() {
    rva_out.Reset();
    data_out.Reset();
    done.Reset();
    pe_start.Reset();
    wait();

    //unsigned i = 0;
    while (1) {
      if (rva_out.PopNB(rva_out_dest)) {
        cout << sc_time_stamp() << " Dest rva_out read data" << " 	 " << endl;
        for (int i = 0; i < spec::kNumVectorLanes; i++) {
          AdpfloatType<8,3> tmp(rva_out_dest.data[i]);
          //cout << tmp.to_float(2) << endl; //XXX check adativefloat bias value 
        }
      }
      wait(1);
    } //while
  } //RVA_Out

  void PopDataOut() {
   wait();
   while (1) {
     if (data_out.PopNB(data_out_dest)) {
        cout << sc_time_stamp() << " Design data_out result" << " 	 " << endl;
        Output.clear();
        for (int i = 0; i < spec::kNumVectorLanes; i++) {
          AdpfloatType<8,3> tmp(data_out_dest.data[i]);
          Output.push_back(tmp.to_float(2));
        }
        
        double diff = 0.0;
        double max_diff = 0.0;
        for (int i = 0; i < Output.size(); i++) {
          diff = abs(Output[i] - golden_y[golden_y_offset + i]);
          //cout << "i and golden_y_offset = " << i << " " << golden_y_offset << endl;
          cout << "Output = " << Output[i] << " golden_y = " << golden_y[golden_y_offset + i] << " diff = " << diff << endl;
          if (diff > max_diff) {
            max_diff = diff;
          }
        }
        golden_y_offset += Output.size();

        if (max_diff > 0.2) {
          std::cout << "TESTBENCH FAIL: max diff = " << max_diff << " greater than 0.2" << std::endl;
          sc_assert(false);
        }
        else {
          std::cout << "TESTBENCH PASS: max diff = " << max_diff << " less than 0.2" << std::endl;
        }
        done_PopDataOut = true;
     }
     wait(); 
   } // while
  } //PopDataOut

  void PopDone() {
   wait();
   while (1) {
     if (done.PopNB(done_dest)) {
        cout << sc_time_stamp() << " Done signal issued!!!" << endl;
        done_PopDone = true;
     }
     wait(); 
   } // while
  } //PopDone

  void PopPEStart() {
   wait();
   while (1) {
     if (pe_start.PopNB(pe_start_dest)) {
        cout << sc_time_stamp() << " PE_start signal issued!!!" << endl;
        done_PopPEStart = true;
     }
     wait(); 
   } // while
  } //PopPEStart

  void SimStop() {
    wait();
    while(1) {
      wait();
      if (done_PopDataOut && done_PopDone && done_PopPEStart) {
        std::cout << "All signals received" << std::endl;
        sc_stop();
      }
    }
  }


}; //SC MODULE Dest

int Dest::golden_y_offset = 0;

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

    dest.clk(clk);
    dest.rst(rst);
    dest.rva_out(rva_out);
    dest.data_out(data_out);
    dest.pe_start(pe_start);
    dest.done(done);

    SC_THREAD(run); 
  } //testbench
  
  void run(){
    wait(2, SC_NS );
    std::cout << "@" << sc_time_stamp() <<" Asserting reset" << std::endl;
    rst.write(false);
    wait(2, SC_NS );
    rst.write(true);
    std::cout << "@" << sc_time_stamp() <<" De-Asserting reset" << std::endl;
    wait(100000, SC_NS );
    std::cout << "@" << sc_time_stamp() <<" Timeout Error!" << std::endl;
    sc_assert(false);
    sc_stop();
  }
}; //SC Module testbench

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
