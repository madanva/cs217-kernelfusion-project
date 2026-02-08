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

// ============================================================================ 
// Test Application for design_top on AWS F2 FPGA
// ============================================================================ 
// This application tests the design_top AXI interface on an AWS F2 instance.
// It mimics the SystemVerilog testbench (verif/tests/design_top_base_test.sv)
// to perform basic AXI write and read operations.
//
// Test steps:
//  (a) Initialize FPGA and PCI interface.
//  (b) Perform a series of AXI write operations.
//  (c) Perform a series of AXI read operations and verify the data.
// ============================================================================ 

#include "design_top.h"
#include <fpga_mgmt.h>
#include <fpga_pci.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// #define DEBUG

// ============================================================================ 
// Low-level MMIO Functions
// ============================================================================ 

int ocl_wr32(int bar_handle, uint16_t addr, uint32_t data) {
  if (fpga_pci_poke(bar_handle, addr, data)) {
    fprintf(stderr, "ERROR: MMIO write failed at addr=0x%04x\n", addr);
    return 1;
  }
  return 0;
}

int ocl_rd32(int bar_handle, uint16_t addr, uint32_t* data) {
  if (fpga_pci_peek(bar_handle, addr, data)) {
    fprintf(stderr, "ERROR: MMIO read failed at addr=0x%04x\n", addr);
    return 1;
  }
  return 0;
}

// ============================================================================ 
// Top-level AXI Interface Functions
// ============================================================================ 

/**
 * @brief Send an AXI write command to the FPGA.
 *
 * Mimics the 'top_write' task in the SystemVerilog testbench.
 */
int top_write(int bar_handle, const AxiWriteCommand* write_command) {
    uint64_t transfer_addr_full = ((uint64_t)write_command->addr << 10);
    uint32_t transfer_addr[LOOP_TOP_AXI_AW] = {0};

    transfer_addr[0] = transfer_addr_full & 0xFFFFFFFF;
    transfer_addr[1] = (transfer_addr_full >> 32) & 0x3FFFF; // 18 bits

    // Write address to AW channel
    for (int i = 0; i < LOOP_TOP_AXI_AW; i++) {
        if (ocl_wr32(bar_handle, ADDR_TOP_AXI_AW_START + i * 4, transfer_addr[i])) {
            return 1;
        }
    }

    usleep(10); // Small delay

    // Write data to W channel
    uint32_t transfer_data[LOOP_TOP_AXI_W] = {0};
    transfer_data[0] = write_command->data[0];
    transfer_data[1] = write_command->data[1];
    transfer_data[2] = write_command->data[2];
    transfer_data[3] = write_command->data[3];
    transfer_data[4] = 0x1FFFF; // Strobe

    for (int i = 0; i < LOOP_TOP_AXI_W; i++) {
        if (ocl_wr32(bar_handle, ADDR_TOP_AXI_W_START + i * 4, transfer_data[i])) {
            return 1;
        }
    }
    return 0;
}


/**
 * @brief Send an AXI read command and retrieve data from the FPGA.
 *
 * Mimics the 'top_read' task in the SystemVerilog testbench.
 */
int top_read(int bar_handle, AxiReadCommand* read_command) {
    uint64_t transfer_addr_full = ((uint64_t)read_command->addr << 10);
    uint32_t transfer_addr[LOOP_TOP_AXI_AR] = {0};

    transfer_addr[0] = transfer_addr_full & 0xFFFFFFFF;
    transfer_addr[1] = (transfer_addr_full >> 32) & 0x3FFFF; // 18 bits

    // Write address to AR channel
    for (int i = 0; i < LOOP_TOP_AXI_AR; i++) {
      //data sent
        if (ocl_wr32(bar_handle, ADDR_TOP_AXI_AR_START + i * 4, transfer_addr[i])) {
            return 1;
        }
    }

    usleep(10); // Small delay

    // Read data from R channel
    uint32_t transfer_data[LOOP_TOP_AXI_R] = {0};
    for (int i = 0; i < LOOP_TOP_AXI_R; i++) {
        if (ocl_rd32(bar_handle, ADDR_TOP_AXI_R_START + i * 4, &transfer_data[i])) {
            return 1;
        }
    }

    // Unpack data (141 bits total, data is in bits 137:10)
    read_command->data[0] = (transfer_data[0] >> 10) | ((transfer_data[1] & 0x3FF) << 22);
    read_command->data[1] = (transfer_data[1] >> 10) | ((transfer_data[2] & 0x3FF) << 22);
    read_command->data[2] = (transfer_data[2] >> 10) | ((transfer_data[3] & 0x3FF) << 22);
    read_command->data[3] = (transfer_data[3] >> 10) | ((transfer_data[4] & 0x3FF) << 22);


    // Verify data
    if (memcmp(read_command->data, read_command->expected_read_data, sizeof(read_command->data)) != 0) {
        fprintf(stderr, "Read data vs expected data mismatch!\n");
        fprintf(stderr, "  Address: 0x%X\n", read_command->addr);
        fprintf(stderr, "  Read:      0x%08X_%08X_%08X_%08X\n", read_command->data[3], read_command->data[2], read_command->data[1], read_command->data[0]);
        fprintf(stderr, "  Expected:  0x%08X_%08X_%08X_%08X\n", read_command->expected_read_data[3], read_command->expected_read_data[2], read_command->expected_read_data[1], read_command->expected_read_data[0]);
        return 1; // Mismatch
    } else {
        printf("Read value matches the expected = 0x%08X_%08X_%08X_%08X at 0x%X\n", read_command->data[3], read_command->data[2], read_command->data[1], read_command->data[0], read_command->addr);
    }

    return 0; // Success
}


// ============================================================================ 
// Main Test Application
// ============================================================================ 

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <slot_id>\n", argv[0]);
    return 1;
  }

  int slot_id    = atoi(argv[1]);
  int bar_handle = -1;
  int rc         = 0;

  // ========================================================================= 
  // 1. Initialization and Attachment
  // ========================================================================= 
  if (fpga_mgmt_init() != 0) {
    fprintf(stderr, "Failed to initialize fpga_mgmt\n");
    return 1;
  }

  if (fpga_pci_attach(slot_id, FPGA_APP_PF, APP_PF_BAR0, 0, &bar_handle)) {
    fprintf(stderr, "fpga_pci_attach failed\n");
    return 1;
  }
  printf("---- System Initialization (bar_handle: %d) ----\n", bar_handle);

  // ========================================================================= 
  // Test Sequence
  // ========================================================================= 
  printf("\n---- Running AXI Write/Read Test ----\n");

  AxiWriteCommand write_cmd = {
      .addr = 0x33500000,
      .data = {0x9EE3E635, 0x584169B2, 0xA0A882BF, 0xD4C04352}};

  AxiReadCommand read_cmd = {
      .addr = 0x33500000,
      .expected_read_data = {0x9EE3E635, 0x584169B2, 0xA0A882BF, 0xD4C04352}};

  // Perform Write
  if (top_write(bar_handle, &write_cmd)) {
      rc = 1;
      goto cleanup;
  }

  // Perform Read and Verify
  if (top_read(bar_handle, &read_cmd)) {
      rc = 1;
      goto cleanup;
  }

  // ========================================================================= 
  // Test Complete
  // ========================================================================= 
  printf("\n---- TEST %s ----\n", (rc == 0) ? "PASSED" : "FAILED");

cleanup:
  if (bar_handle != -1) {
    fpga_pci_detach(bar_handle);
  }

  return rc;
}