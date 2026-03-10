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
    uint32_t transfer_data[LOOP_TOP_AXI_R] = {0};

    transfer_addr[0] = transfer_addr_full & 0xFFFFFFFF;
    transfer_addr[1] = (transfer_addr_full >> 32) & 0x3FFFF; // 18 bits

    // Write address to AR channel
    for (int i = 0; i < LOOP_TOP_AXI_AR; i++) {
        if (ocl_wr32(bar_handle, ADDR_TOP_AXI_AR_START + i * 4, transfer_addr[i])) {
            return 1;
        }
    }

    usleep(10); // Small delay

    // Read data from R channel
    for (int i = 0; i < LOOP_TOP_AXI_R; i++) {
        transfer_data[i] = 0;
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
        fprintf(stderr, "\nRead data vs expected data mismatch!\n");
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
// Attention Host Driver Functions
// ============================================================================

// AXI address encoding for GBPartition (prefix 0x33):
//   tag 0x4 = GBCore config, tag 0x5 = GBCore SRAM data
//   tag 0xD = Attention config, tag 0x0 = start triggers
// Address format: 0x33 | tag(4b) | local_index(16b) | 0
// SRAM addr N -> 0x3350_0000 + (N << 4)

#define ADDR_GB_SRAM(phys_addr)    (0x33500000 | ((phys_addr) << 4))
#define ADDR_GB_CONFIG             0x33400010
#define ADDR_ATTN_CONFIG           0x330D0010
#define ADDR_ATTN_PERF             0x330D0020
#define ADDR_ATTN_START            0x33000030

// N=16 SRAM layout: Q@0-15, K@16-31, V@32-47, O@48-63
#define SRAM_BASE_Q   0
#define SRAM_BASE_K   16
#define SRAM_BASE_V   32
#define SRAM_BASE_O   48

/**
 * @brief Write a 128-bit vector to GBCore SRAM at physical address.
 */
int attn_write_sram(int bar_handle, uint16_t phys_addr, const uint32_t data[4]) {
  AxiWriteCommand cmd = {ADDR_GB_SRAM(phys_addr), {data[0], data[1], data[2], data[3]}};
  return top_write(bar_handle, &cmd);
}

/**
 * @brief Read a 128-bit vector from GBCore SRAM at physical address.
 */
int attn_read_sram(int bar_handle, uint16_t phys_addr, uint32_t data_out[4]) {
  AxiReadCommand cmd;
  cmd.addr = ADDR_GB_SRAM(phys_addr);
  memset(cmd.data, 0, sizeof(cmd.data));
  // Set expected to match anything (we check manually after)
  memset(cmd.expected_read_data, 0, sizeof(cmd.expected_read_data));

  // We can't use top_read directly since it checks expected_read_data.
  // Instead, do the raw AXI read sequence.
  uint64_t transfer_addr_full = ((uint64_t)cmd.addr << 10);
  uint32_t transfer_addr[LOOP_TOP_AXI_AR] = {0};
  uint32_t transfer_data[LOOP_TOP_AXI_R] = {0};

  transfer_addr[0] = transfer_addr_full & 0xFFFFFFFF;
  transfer_addr[1] = (transfer_addr_full >> 32) & 0x3FFFF;

  for (int i = 0; i < LOOP_TOP_AXI_AR; i++) {
    if (ocl_wr32(bar_handle, ADDR_TOP_AXI_AR_START + i * 4, transfer_addr[i]))
      return 1;
  }
  usleep(10);

  for (int i = 0; i < LOOP_TOP_AXI_R; i++) {
    if (ocl_rd32(bar_handle, ADDR_TOP_AXI_R_START + i * 4, &transfer_data[i]))
      return 1;
  }

  // Unpack (same as top_read: data is in bits [137:10])
  data_out[0] = (transfer_data[0] >> 10) | ((transfer_data[1] & 0x3FF) << 22);
  data_out[1] = (transfer_data[1] >> 10) | ((transfer_data[2] & 0x3FF) << 22);
  data_out[2] = (transfer_data[2] >> 10) | ((transfer_data[3] & 0x3FF) << 22);
  data_out[3] = (transfer_data[3] >> 10) | ((transfer_data[4] & 0x3FF) << 22);
  return 0;
}

/**
 * @brief Configure GBCore address mapping for 4 memory regions.
 *
 * GBCore config register layout (128 bits):
 *   mem0: num_vector[7:0], base[31:16]
 *   mem1: num_vector[39:32], base[63:48]
 *   mem2: num_vector[71:64], base[95:80]
 *   mem3: num_vector[103:96], base[127:112]
 */
int attn_configure_gbcore(int bar_handle,
                          uint16_t base0, uint16_t base1,
                          uint16_t base2, uint16_t base3) {
  // Each region: num_vector=1, base=baseN
  // Bit layout: [7:0]=num_vec, [31:16]=base for each 32-bit slot
  uint32_t data[4];
  data[0] = 1 | ((uint32_t)base0 << 16);  // mem0 (Q)
  data[1] = 1 | ((uint32_t)base1 << 16);  // mem1 (K)
  data[2] = 1 | ((uint32_t)base2 << 16);  // mem2 (V)
  data[3] = 1 | ((uint32_t)base3 << 16);  // mem3 (O)
  AxiWriteCommand cmd = {ADDR_GB_CONFIG, {data[0], data[1], data[2], data[3]}};
  return top_write(bar_handle, &cmd);
}

/**
 * @brief Configure attention module parameters.
 *
 * Config register layout (128 bits, matches AttentionSpec.h ConfigWrite):
 *   [0]     is_valid
 *   [10:8]  q_memory_index
 *   [18:16] k_memory_index
 *   [26:24] v_memory_index
 *   [34:32] o_memory_index
 *   [55:48] seq_len
 *   [71:64] head_dim
 *   [87:80] num_tiles
 */
int attn_configure(int bar_handle, uint8_t seq_len, uint8_t num_tiles) {
  uint32_t data[4] = {0, 0, 0, 0};
  // word0 bits[0]: is_valid=1, bits[10:8]: q_mem=0, bits[18:16]: k_mem=1, bits[26:24]: v_mem=2
  data[0] = 1 | (0 << 8) | (1 << 16) | (2 << 24);
  // word1 bits[2:0]: o_mem=3, bits[23:16]: seq_len
  data[1] = 3 | ((uint32_t)seq_len << 16);
  // word2 bits[7:0]: head_dim=16, bits[23:16]: num_tiles
  data[2] = 16 | ((uint32_t)num_tiles << 16);
  AxiWriteCommand cmd = {ADDR_ATTN_CONFIG, {data[0], data[1], data[2], data[3]}};
  return top_write(bar_handle, &cmd);
}

/**
 * @brief Trigger attention computation start.
 */
int attn_start(int bar_handle) {
  AxiWriteCommand cmd = {ADDR_ATTN_START, {0, 0, 0, 0}};
  return top_write(bar_handle, &cmd);
}

/**
 * @brief Wait for attention completion by polling interrupt cycles counter.
 *
 * The interrupt counter increments while the interrupt signal is high.
 * We poll until it stops changing (computation complete).
 */
int attn_wait_done(int bar_handle, uint32_t* cycles_out) {
  uint32_t prev_cycles = 0;
  uint32_t curr_cycles = 0;
  int stable_count = 0;

  for (int attempt = 0; attempt < 10000; attempt++) {
    if (ocl_rd32(bar_handle, ADDR_TOP_INTERRUPT, &curr_cycles))
      return 1;

    if (curr_cycles > 0 && curr_cycles == prev_cycles) {
      stable_count++;
      if (stable_count >= 3) {
        if (cycles_out) *cycles_out = curr_cycles;
        return 0;
      }
    } else {
      stable_count = 0;
    }
    prev_cycles = curr_cycles;
    usleep(100);
  }

  fprintf(stderr, "ERROR: Attention timed out (last interrupt_cycles=%u)\n", curr_cycles);
  if (cycles_out) *cycles_out = curr_cycles;
  return 1;
}

/**
 * @brief Read attention performance counters via AXI.
 *
 * Perf counter register layout (128 bits, ConfigRead index 0x02):
 *   [31:0]  cycle_count
 *   [63:32] gb_read_count
 *   [95:64] gb_write_count
 */
int attn_read_perf(int bar_handle, uint32_t* cycles,
                   uint32_t* reads, uint32_t* writes) {
  uint32_t data[4] = {0};
  if (attn_read_sram(bar_handle, 0, data)) return 1; // dummy, we need raw read

  // Use the raw AXI read for the perf counter address
  uint64_t transfer_addr_full = ((uint64_t)ADDR_ATTN_PERF << 10);
  uint32_t transfer_addr[LOOP_TOP_AXI_AR] = {0};
  uint32_t transfer_data[LOOP_TOP_AXI_R] = {0};

  transfer_addr[0] = transfer_addr_full & 0xFFFFFFFF;
  transfer_addr[1] = (transfer_addr_full >> 32) & 0x3FFFF;

  for (int i = 0; i < LOOP_TOP_AXI_AR; i++) {
    if (ocl_wr32(bar_handle, ADDR_TOP_AXI_AR_START + i * 4, transfer_addr[i]))
      return 1;
  }
  usleep(10);

  for (int i = 0; i < LOOP_TOP_AXI_R; i++) {
    if (ocl_rd32(bar_handle, ADDR_TOP_AXI_R_START + i * 4, &transfer_data[i]))
      return 1;
  }

  uint32_t unpacked[4];
  unpacked[0] = (transfer_data[0] >> 10) | ((transfer_data[1] & 0x3FF) << 22);
  unpacked[1] = (transfer_data[1] >> 10) | ((transfer_data[2] & 0x3FF) << 22);
  unpacked[2] = (transfer_data[2] >> 10) | ((transfer_data[3] & 0x3FF) << 22);
  unpacked[3] = (transfer_data[3] >> 10) | ((transfer_data[4] & 0x3FF) << 22);

  if (cycles) *cycles = unpacked[0];
  if (reads)  *reads  = unpacked[1];
  if (writes) *writes = unpacked[2];
  return 0;
}

/**
 * @brief Generate deterministic test data (int8 range, matches testbench pattern).
 */
void attn_generate_test_data(uint32_t vectors[][4], int count, int seed) {
  for (int i = 0; i < count; i++) {
    uint32_t w[4] = {0};
    for (int j = 0; j < 16; j++) {
      int8_t val = (int8_t)(((seed + i * 16 + j) * 7 + 13) % 256 - 128);
      // Pack int8 into 128-bit vector (16 x 8-bit values in 4 x 32-bit words)
      int word_idx = j / 4;
      int byte_idx = j % 4;
      w[word_idx] |= ((uint32_t)(uint8_t)val) << (byte_idx * 8);
    }
    memcpy(vectors[i], w, sizeof(w));
  }
}

// ============================================================================
// Attention FPGA Test
// ============================================================================

int attn_fpga_test(int bar_handle, int seq_len) {
  int rc = 0;
  int num_tiles = (seq_len + 15) / 16;

  printf("\n---- Attention FPGA Test (N=%d) ----\n", seq_len);

  // Step 1: Configure GBCore address mapping
  printf("  Configuring GBCore address mapping...\n");
  if (attn_configure_gbcore(bar_handle,
                            SRAM_BASE_Q, SRAM_BASE_K,
                            SRAM_BASE_V, SRAM_BASE_O)) {
    fprintf(stderr, "ERROR: GBCore config failed\n");
    return 1;
  }
  usleep(10);

  // Step 2: Generate and write Q, K, V data to GBCore SRAM
  uint32_t q_data[64][4], k_data[64][4], v_data[64][4];
  attn_generate_test_data(q_data, seq_len, 42);
  attn_generate_test_data(k_data, seq_len, 137);
  attn_generate_test_data(v_data, seq_len, 271);

  printf("  Writing Q data (SRAM %d-%d)...\n", SRAM_BASE_Q, SRAM_BASE_Q + seq_len - 1);
  for (int i = 0; i < seq_len; i++) {
    if (attn_write_sram(bar_handle, SRAM_BASE_Q + i, q_data[i])) { rc = 1; break; }
    usleep(10);
  }

  printf("  Writing K data (SRAM %d-%d)...\n", SRAM_BASE_K, SRAM_BASE_K + seq_len - 1);
  for (int i = 0; i < seq_len; i++) {
    if (attn_write_sram(bar_handle, SRAM_BASE_K + i, k_data[i])) { rc = 1; break; }
    usleep(10);
  }

  printf("  Writing V data (SRAM %d-%d)...\n", SRAM_BASE_V, SRAM_BASE_V + seq_len - 1);
  for (int i = 0; i < seq_len; i++) {
    if (attn_write_sram(bar_handle, SRAM_BASE_V + i, v_data[i])) { rc = 1; break; }
    usleep(10);
  }

  if (rc) {
    fprintf(stderr, "ERROR: SRAM data write failed\n");
    return rc;
  }

  // Step 3: Configure attention module
  printf("  Configuring attention (seq_len=%d, num_tiles=%d)...\n", seq_len, num_tiles);
  if (attn_configure(bar_handle, seq_len, num_tiles)) {
    fprintf(stderr, "ERROR: Attention config failed\n");
    return 1;
  }
  usleep(10);

  // Step 4: Trigger attention start
  printf("  Starting attention computation...\n");
  if (attn_start(bar_handle)) {
    fprintf(stderr, "ERROR: Attention start failed\n");
    return 1;
  }

  // Step 5: Wait for completion
  uint32_t interrupt_cycles = 0;
  if (attn_wait_done(bar_handle, &interrupt_cycles)) {
    rc = 1;
  }
  printf("  Attention complete. Interrupt cycles = %u\n", interrupt_cycles);

  // Step 6: Read performance counters
  uint32_t perf_cycles = 0, perf_reads = 0, perf_writes = 0;
  if (attn_read_perf(bar_handle, &perf_cycles, &perf_reads, &perf_writes)) {
    fprintf(stderr, "WARNING: Could not read perf counters\n");
  } else {
    printf("  Perf counters: cycles=%u, gb_reads=%u, gb_writes=%u\n",
           perf_cycles, perf_reads, perf_writes);
  }

  // Step 7: Read output from GBCore SRAM
  printf("  Reading output (SRAM %d-%d)...\n", SRAM_BASE_O, SRAM_BASE_O + seq_len - 1);
  for (int i = 0; i < seq_len; i++) {
    uint32_t output[4] = {0};
    if (attn_read_sram(bar_handle, SRAM_BASE_O + i, output)) {
      fprintf(stderr, "ERROR: Output read failed at row %d\n", i);
      rc = 1;
      break;
    }
    printf("  Output[%2d] = 0x%08X_%08X_%08X_%08X\n",
           i, output[3], output[2], output[1], output[0]);
  }

  // Step 8: FPGA timing measurement
  // interrupt_cycles counts at 125 MHz (8ns per cycle)
  double fpga_time_us = interrupt_cycles * 0.008;
  printf("\n  FPGA wall-clock time: %.2f us (%u cycles @ 125 MHz)\n",
         fpga_time_us, interrupt_cycles);

  printf("\n---- Attention FPGA Test: %s ----\n", (rc == 0) ? "PASSED" : "FAILED");
  return rc;
}

// ============================================================================
// Main Test Application
// ============================================================================

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("Usage: %s <slot_id> [attn]\n", argv[0]);
    printf("  slot_id: FPGA slot number (usually 0)\n");
    printf("  attn:    run attention test instead of base test\n");
    return 1;
  }

  int slot_id    = atoi(argv[1]);
  int run_attn   = (argc >= 3 && strcmp(argv[2], "attn") == 0);
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

  if (run_attn) {
    // =====================================================================
    // Attention Test
    // =====================================================================
    rc = attn_fpga_test(bar_handle, 16);
  } else {
    // =====================================================================
    // Original Base Test Sequence
    // =====================================================================
    printf("\n---- Running AXI Write/Read Test ----\n");

    AxiWriteCommand write_commands[] = {
        {0x33500000, {0x9EE3E635, 0x584169B2, 0xA0A882BF, 0xD4C04352}},
        {0x34500000, {0x88E1D68C, 0x6BD421D7, 0x5C7F3202, 0xC7427867}},
        {0x34500010, {0x5EC23966, 0xA174272E, 0x21E7A2FD, 0xD0319B6C}},
        {0x34500020, {0x178B1B85, 0xA331DDE2, 0xB8E9DD33, 0x5781547C}},
        {0x34500030, {0x8D22BBEB, 0x4E92D920, 0x04BCB961, 0x4C8C4B83}},
        {0x34500040, {0xC5BFA479, 0x0DC7A487, 0xA9D9B720, 0x67AD5414}},
        {0x34500050, {0x4A09CF2D, 0x0292B32C, 0xD70083F7, 0x69AB46F7}},
        {0x34500060, {0x27208FCB, 0xD103A7F4, 0x9B261E3F, 0x161F6574}},
        {0x34500070, {0xE1356000, 0xED6A7A4A, 0xB2819ED0, 0xAABCB5EF}},
        {0x34500080, {0xA9695EE4, 0xC59C9EC4, 0x5D2D4CDA, 0xF6D7D941}},
        {0x34500090, {0x3E0DFD81, 0x7F151973, 0xF78E9E7F, 0x17899ACB}},
        {0x345000A0, {0x4E3AD635, 0xACC64781, 0x69A343A4, 0xFCFD96D1}},
        {0x345000B0, {0x58371BA5, 0x8582459D, 0xE065D484, 0x5C0F148D}},
        {0x345000C0, {0x1E5515A8, 0xA96684FC, 0xB30AE0F6, 0xCC77DBF7}},
        {0x345000D0, {0xDF439320, 0xC97FD011, 0x13F2CD9D, 0xC5AA4918}},
        {0x345000E0, {0x2C4EE908, 0x520BE5B5, 0x72129DD4, 0xB8E6F69A}},
        {0x345000F0, {0xAE2D9CD6, 0x679295D0, 0xDFD4E551, 0x38B305DE}},
        {0x34400010, {0x1, 0x101, 0x0, 0x0}},
        {0x34400020, {0x100, 0x0, 0x0, 0x0}},
        {0x34800010, {0x3020001, 0x1, 0x0, 0x0}},
        {0x34800020, {0x40B030, 0x0, 0x0, 0x0}},
        {0x33400010, {0x1, 0x0, 0x0, 0x0}},
        {0x33700010, {0x1, 0x1010100, 0x10001, 0x0}},
        {0x33000010, {0x0, 0x0, 0x0, 0x0}},
        {0x33400010, {0x10001, 0x0, 0x0, 0x0}},
        {0x33500010, {0x761D3767, 0x5D0340C6, 0x3652115C, 0x298E1EFC}},
        {0x33C00010, {0x101, 0x10000, 0x1, 0x0}},
        {0x33000020, {0x0, 0x0, 0x0, 0x0}},
        {0x345000F0, {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF}},
        {0x345000F0, {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF}},
        {0x345000F0, {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF}},
        {0x345000F0, {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF}},
        {0x345000F0, {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF}},
        {0x345000F0, {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF}},
        {0x345000F0, {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF}},
        {0x345000F0, {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF}},
        {0x345000F0, {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF}},
        {0x345000F0, {0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF}},
    };

    AxiReadCommand read_commands[] = {
        {0x33500010, {0}, {0x8000003, 0x1000000, 0x1, 0x0}},
        {0x33500000, {0}, {0x10101010, 0x10101010, 0x10101010, 0x10101010}},
        {0x34600000, {0}, {0x9EE3E635, 0x584169B2, 0xA0A882BF, 0xD4C04352}},
     };

    int num_write_commands = sizeof(write_commands) / sizeof(AxiWriteCommand);
    for (int i = 0; i < num_write_commands; i++) {
        if (top_write(bar_handle, &write_commands[i])) {
            rc = 1;
        }
        usleep(10);
    }

    int num_read_commands = sizeof(read_commands) / sizeof(AxiReadCommand);
    for (int i = 0; i < num_read_commands; i++) {
        if (top_read(bar_handle, &read_commands[i])) {
            rc = 1;
        }
        usleep(10);
    }

    // Read Interrupt Cycles Counter
    uint32_t interrupt_cycles = 0;
    printf("\n---- Reading Interrupt Cycles Counter ----\n");
    if (ocl_rd32(bar_handle, ADDR_TOP_INTERRUPT, &interrupt_cycles)) {
        rc = 1;
    }

    printf("Interrupt cycles = %u\n", interrupt_cycles);
    if (interrupt_cycles <= 10) {
        fprintf(stderr, "ERROR: Interrupt cycles lesser than expected! Interrupt cycles = %u\n", interrupt_cycles);
        rc = 1;
    }
  }

  // =========================================================================
  // Test Complete
  // =========================================================================
  printf("\n---- TEST %s ----\n", (rc == 0) ? "PASSED" : "FAILED");

  if (bar_handle != -1) {
    fpga_pci_detach(bar_handle);
  }

  return rc;
}
