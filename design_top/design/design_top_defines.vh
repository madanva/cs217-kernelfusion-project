// ============================================================================
// Amazon FPGA Hardware Development Kit
//
// Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
//
// Licensed under the Amazon Software License (the "License"). You may not use
// this file except in compliance with the License. A copy of the License is
// located at
//
//    http://aws.amazon.com/asl/
//
// or in the "license" file accompanying this file. This file is distributed on
// an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or
// implied. See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================
`ifndef design_top_DEFINES
`define design_top_DEFINES

        // Put module name of the CL design here. This is used to instantiate in top.sv
`define CL_NAME design_top

`define DDR_A_ABSENT
`define DDR_B_ABSENT


`endif

// ============================================================================
// GBModule Interface Parameters
// ============================================================================
// These parameters match the HLS-generated GBModule interface from
// design_top/design/concat_GBModule.v

// Vector and data widths
localparam int kVectorSize = 16;
localparam int kAdpfloatWordWidth = 8;
localparam int kVectorWidth = kVectorSize * kAdpfloatWordWidth; // 128 bits

// RVA interface widths (from AxiSpec.h)
// rva_in_dat: data(128) + addr(24) + wstrb(16) + rw(1) = 169 bits
localparam int kRVADataWidth = 128;
localparam int kRVAAddrWidth = 24;
localparam int kRVAWstrbWidth = 16;
localparam int kRVAInWidth = 169;

// rva_out_dat: 128 bits
localparam int kRVAOutWidth = 128;

// Legacy compatibility (for existing infrastructure)
localparam int kIntWordWidth = 16;
localparam int kNumVectorLanes = 16;
localparam int kActWordWidth = 32;
localparam int kAccumWordWidth = 2*kIntWordWidth + kNumVectorLanes - 1;
localparam int kActNumFrac = 24;

localparam int WIDTH_AXI = 32;
localparam int ADDR_WIDTH_OCL = 16;

// Transfer Counter
localparam int ADDR_TX_COUNTER_EN = 16'h0400;         // Write 
localparam int ADDR_TX_COUNTER_READ = 16'h0400;       // Read
localparam int ADDR_COMPUTE_COUNTER_READ = 16'h0404;  // Read

// Start enable
localparam int ADDR_START_CFG = 16'h0404;             //Write

// RVA IN Port (Input) - updated for GBModule 169-bit interface
localparam int DATA_WIDTH_RVA_IN = kRVADataWidth;                             // 128 bits
localparam int ADDR_WIDTH_RVA_IN = kRVAAddrWidth;                             // 24 bits
localparam int WIDTH_RVA_IN = kRVAInWidth;                                    // 169 bits
localparam int LOOP_RVA_IN =  (int((WIDTH_RVA_IN + 31)/ 32));                 // 6 words
localparam int ADDR_RVA_IN_START = 16'h408;

// RVA Out Port (Output)
localparam int WIDTH_RVA = kRVAOutWidth;                                      // 128 bits
localparam int LOOP_RVA_OUT = int((WIDTH_RVA + 31) / 32);                     // 4 words
localparam int ADDR_RVA_OUT_START = 16'h44C;
