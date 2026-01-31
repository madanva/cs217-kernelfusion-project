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

#include <iostream>

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
 public:
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

    std::vector<std::vector<double>> W(spec::kNumVectorLanes, std::vector<double>(spec::kNumVectorLanes));

    spec::VectorType act_vec;

    AdpfloatType<8, 3> adpfloat_tmp;
    spec::AdpfloatBiasType adpbias_input = 2;
    wait();

    //GBControl AXI GBControlConfig
    // is_valid=1, mode=0 (unidirectional), is_rnn=1, memory_index=0, output_memory_index=0,
    // num_vector= 1, num_timestep=16,
    // adpbias_act=2, adpbias_atten=0, adpbias_beta = 0, adpbias_gamma=0
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("00_00_00_02_00_00_00_10_00_01_00_00_00_01_00_01");
    rva_in_src.addr = set_bytes<3>("70_00_10");  // last 4 bits never used
    cout << "    WRITE GBControlConfig: " << rva_in_src.data << " @ " << rva_in_src.addr << endl;  
    rva_in.Push(rva_in_src);
    wait();

    //GBCore LargeBuffer Config
    // num_vector_large for kMaxNumManagers=0 is 16, base_large for kMaxNumManagers=0 is 0
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_10");
    rva_in_src.addr = set_bytes<3>("40_00_10");  // last 4 bits never used
    cout << "    WRITE GBCore LargeBufferConfig: " << rva_in_src.data << " @ " << rva_in_src.addr << endl;  
    rva_in.Push(rva_in_src);
    wait();

    //Storing Activations in LargeBuffer
    for (int i = 0; i < 16; i++) { // 16 vectors for our 16x16 matrix
        for (int k=0; k < spec::kNumVectorLanes; k++) { //k --> scalar_index
            adpfloat_tmp.Reset();
            double random_val = ((double)rand() / RAND_MAX * 2.0 - 1.0);
            W[i][k] = random_val;
            cout << "Value sent: " << random_val << endl;
            adpfloat_tmp.set_value(random_val, adpbias_input);
            act_vec[k] = adpfloat_tmp.to_rawbits();
        }
        rva_in_src.rw = 1;
        rva_in_src.data = act_vec.to_rawbits();
        rva_in_src.addr = 0x500000 + i * 16; // Simple contiguous address for each 16-byte vector
        rva_in.Push(rva_in_src);
        if (i == 15) {
            cout << "Pushed last input vector." << endl;
        }
        wait();
    }
    wait();

    // Golden model calculation for layer normalization

    

    //GBControl AXI GBControlConfig
    // is_valid=1, mode=0, sendback=0, memory_index=0, output_memory_index=0,
    // num_vector= 1, num_timestep=16,
    // adpbias_act=2, adpbias_atten=0, adpbias_beta = 0, adpbias_gamma=1
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("01_00_00_02_00_00_00_10_00_01_00_00_00_00_00_01");
    rva_in_src.addr = set_bytes<3>("70_00_10");  // last 4 bits never used
    rva_in.Push(rva_in_src);
    wait();

    //send GBControl start signal
    rva_in_src.rw = 1;
    rva_in_src.data = set_bytes<16>("00_00_00_00_00_00_00_00_00_00_00_00_00_00_00_00");
    rva_in_src.addr = set_bytes<3>("00_00_10");
    rva_in.Push(rva_in_src);
    wait();
 
  } // layernorm_run()

}; //SC MODULE Source

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

    //unsigned i = 0;
    while (1) {
      if (rva_out.PopNB(rva_out_dest)) {
        cout << sc_time_stamp() << " Dest rva_out read data" << " \t " << endl;
        for (int i = 0; i < spec::kNumVectorLanes; i++) {
          AdpfloatType<8,3> tmp(rva_out_dest.data[i]);
          cout << tmp.to_float(2) << endl; //XXX check adativefloat bias value 
        }
      }
      wait(1);
    } //while
  } //RVA_Out

  void PopDataOut() {
   wait();
   while (1) {
     if (data_out.PopNB(data_out_dest)) {
        cout << sc_time_stamp() << " Design data_out result" << " \t " << endl;
        for (int i = 0; i < spec::kNumVectorLanes; i++) {
          AdpfloatType<8,3> tmp(data_out_dest.data[i]);
          cout << tmp.to_float(2) << endl; //XXX check adativefloat bias value 
        }
     }
     wait(); 
   } // while
  } //PopDataOut

  void PopDone() {
   wait();
   while (1) {
     if (done.PopNB(done_dest)) {
        cout << sc_time_stamp() << " Done signal issued!!!" << " \t " << done_dest << endl;
     }
     wait(); 
   } // while
  } //PopDone

  void PopPEStart() {
   wait();
   while (1) {
     if (pe_start.PopNB(pe_start_dest)) {
        cout << sc_time_stamp() << " PE_start signal issued!!!" << " \t " << pe_start_dest << endl;
     }
     wait(); 
   } // while
  } //PopPEStart

}; //SC MODULE Dest

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
}; //SC Module testbench

int sc_main(int argc, char *argv[]) {
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
