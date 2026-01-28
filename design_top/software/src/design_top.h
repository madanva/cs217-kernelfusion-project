// Copyright 2026 Stanford University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DESIGN_TOP_H
#define DESIGN_TOP_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

// ============================================================================
// GBModule Interface Constants
// ============================================================================

// Vector and data widths
#define kVectorSize 16
#define kAdpfloatWordWidth 8
#define kAdpfloatExpWidth 3
#define kAdpfloatBiasWidth 3

// AXI interface parameters
#define WIDTH_AXI 32
#define ADDR_WIDTH_OCL 16

// RVA interface widths
// rva_in: data(128) + addr(24) + wstrb(16) + rw(1) = 169 bits
#define kRVADataWidth 128
#define kRVAAddrWidth 24
#define kRVAWstrbWidth 16
#define kRVAInWidth 169

// rva_out: 128 bits
#define kRVAOutWidth 128

// Number of 32-bit words for RVA interface
#define LOOP_RVA_IN ((kRVAInWidth + 31) / 32)   // 6 words
#define LOOP_RVA_OUT ((kRVAOutWidth + 31) / 32) // 4 words

// Transfer Counter addresses
#define ADDR_TX_COUNTER_EN 0x0400        // Write
#define ADDR_TX_COUNTER_READ 0x0400      // Read
#define ADDR_COMPUTE_COUNTER_READ 0x0404 // Read

// Start enable address
#define ADDR_START_CFG 0x0404 // Write

// RVA IN Port (Input) - base address for writing RVA messages
#define ADDR_RVA_IN_START 0x0408

// RVA Out Port (Output) - base address for reading RVA responses
// Calculated as: ADDR_RVA_IN_START + LOOP_RVA_IN * 4 + output_port offset
#define ADDR_RVA_OUT_START 0x044C

// Address prefixes for RVA routing (bits [23:20] of address)
#define ADDR_PREFIX_SRAM_CFG 0x3
#define ADDR_PREFIX_GBCORE_CFG 0x4
#define ADDR_PREFIX_GBCORE_DATA 0x5
#define ADDR_PREFIX_NMP_CFG 0xC

// Number of banks in GBCore large buffer
#define kNumBanks 8

// ============================================================================
// Function Prototypes
// ============================================================================

// Low-level MMIO functions
int ocl_wr32(int bar_handle, uint16_t addr, uint32_t data);
int ocl_rd32(int bar_handle, uint16_t addr, uint32_t* data);

// RVA interface functions
void pack_rva_in(
    bool rw,
    uint32_t addr,
    const uint32_t data[LOOP_RVA_OUT],
    uint32_t rva_msg[LOOP_RVA_IN]);
int ocl_rva_wr(int bar_handle, const uint32_t rva_msg[LOOP_RVA_IN]);
int ocl_rva_rd(int bar_handle, uint32_t rva_data[LOOP_RVA_OUT]);

// Configuration helper functions
void make_gbcore_cfg_data(
    uint8_t num_vec, uint16_t base, uint32_t data[LOOP_RVA_OUT]);
void make_nmp_cfg_data(
    uint8_t mode,
    uint8_t mem,
    uint8_t nvec,
    uint16_t ntimestep,
    uint8_t adpbias,
    uint32_t data[LOOP_RVA_OUT]);
uint32_t make_gbcore_data_addr(uint16_t local_index);

// Control functions
int send_start(int bar_handle);
int wait_for_done(int bar_handle);

#endif // DESIGN_TOP_H
