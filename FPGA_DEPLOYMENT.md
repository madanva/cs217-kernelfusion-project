# FPGA Deployment Context

## What We Have

- **3 HLS concat_Top.v files** in `reports/hls_top/{AttnUnfused,AttnPartialFused,AttnFullyFused}/concat_Top.v`
  - All pass scverify at CLK_PERIOD=4ns (250 MHz HLS constraint)
  - FPGA runs at 125 MHz (2x slack)
- **Host driver code** in `design_top/software/src/design_top.{c,h}`
  - `attn_fpga_test(bar_handle, seq_len)` — full end-to-end test
  - Invoke: `sudo ./design_top 0 attn` (attention test) or `sudo ./design_top 0` (original base test)
- **SV testbench** in `design_top/verif/tests/design_top_base_test.sv` — existing base test, tests basic AXI read/write
- **RTL wrapper** in `design_top/design/design_top.sv` — instantiates `Top u_top`, 125 MHz clock, interrupt counter

## File Layout

```
design_top/
  setup.sh              # source this first (sets CL_DIR, CL_DESIGN_NAME, generate_afi func)
  Makefile              # targets: hw_sim, fpga_build, generate_afi, program_fpga, run_fpga_test, clean
  design/
    design_top.sv       # top wrapper, includes concat_Top.v + counter.v
    concat_Top.v        # ← REPLACE THIS with variant's concat_Top.v
    counter.v           # interrupt cycle counter
    design_top_defines.vh
    cl_id_defines.vh
  verif/
    tests/design_top_base_test.sv   # SV testbench (hw_sim runs this)
    scripts/Makefile, Makefile.tests
  software/
    src/design_top.{c,h}   # host driver (compiled by runtime/Makefile)
    runtime/Makefile        # `make design_top` builds the host binary
  build/                   # Vivado build scripts, checkpoints go here
```

## Deployment Steps (Per Variant)

### 0. Environment Setup (on AWS F2 instance)

```bash
cd ~/aws-fpga
git checkout 2f1f343259dcb794adc596a76c31593280fd7c7b
source hdk_setup.sh
source sdk_setup.sh

cd ~/cs217-project/design_top
source setup.sh
```

### 1. Copy RTL — Replace concat_Top.v

```bash
# Pick one variant:
cp reports/hls_top/AttnFullyFused/concat_Top.v design_top/design/concat_Top.v
# OR: AttnUnfused, AttnPartialFused
```

### 2. HW RTL Simulation (VCS, ~minutes)

```bash
cd ~/cs217-project/design_top
make hw_sim
# Log: reports/aws/f2_hw_sim.log.txt
```

This runs VCS on `design_top_base_test.sv`. The existing test writes Q/K/V data, configures GBCore+NMP, triggers computation, reads results. It validates the RTL before FPGA build.

**Important:** The SV testbench sends hardcoded AXI commands. For attention-specific hw_sim, you'd either:
- Modify the SV testbench to send attention commands (see AXI Command Sequence below), OR
- Just verify the base test passes (proves AXI bridge + Top module work) and rely on FPGA host driver for attention testing

### 3. FPGA Build (~2.5 hours)

```bash
cd ~/cs217-project/design_top
make fpga_build
# Strips `// synopsys translate` lines from concat_Top.v, then runs Vivado synthesis+implementation
# Output: build/checkpoints/*.tar (DCP tarball)
```

### 4. Generate AFI

```bash
# Requires S3 bucket env vars (DCP_BUCKET_NAME, DCP_FOLDER_NAME, LOGS_BUCKET_NAME, LOGS_FOLDER_NAME, REGION)
make generate_afi
# Creates generated_afid.sh with AFI and AGFI IDs
```

### 5. Wait for AFI

```bash
make check_afi_available
# Poll until State.Code = "available"
```

### 6. Program FPGA

```bash
make program_fpga
# Clears slot 0, loads AGFI
```

### 7. Run Test

```bash
make run_fpga_test
# Compiles design_top.c, runs: sudo ./design_top 0
# For attention test, modify Makefile or run manually:
cd software/runtime && make design_top
sudo ./design_top 0 attn
# Log: reports/aws/fpga_test.log.txt
```

## AXI Address Map (Host → FPGA)

All addresses go through OCL AXI-Lite (32-bit MMIO). The host serializes 128-bit data as multiple 32-bit writes.

### OCL Register Map

| Register | Address | Direction | Description |
|----------|---------|-----------|-------------|
| AW channel | 0x400-0x404 | Write | 50-bit AXI write address (2 words) |
| W channel | 0x410-0x420 | Write | 145-bit AXI write data (5 words) |
| B channel | 0x430 | Read | 12-bit write response |
| AR channel | 0x440-0x444 | Write | 50-bit AXI read address (2 words) |
| R channel | 0x450-0x460 | Read | 141-bit AXI read data (5 words) |
| Interrupt | 0x570 | Read | 32-bit cycle counter (counts while interrupt high) |

### AXI Address Encoding (inside 50-bit address)

The 32-bit user address is placed in bits [41:10] of the 50-bit AXI address.
Format: `{8'b0, addr[31:0], 10'b0}`

### Top Module Address Decode

Address `0xTTPPLLLL` where:
- `TT` = target partition (0x33 = GBPartition0, 0x34 = GBPartition1, etc.)
- `PP` = tag within partition (decoded by GBModule RVAInRun, bits[23:20])
- `LLLL` = local index (bits[19:4])

**GBPartition0 (0x33) tags:**
| Tag | Address Prefix | Target |
|-----|---------------|--------|
| 0x0 | 0x3300xxxx | Start triggers (local_index 0x1=NMP, 0x2=PE, 0x3=Attention) |
| 0x4 | 0x3340xxxx | GBCore config |
| 0x5 | 0x3350xxxx | GBCore SRAM data |
| 0x7 | 0x3370xxxx | NMP config |
| 0xC | 0x33C0xxxx | PE config |
| 0xD | 0x33D0xxxx | Attention config |

### Attention-Specific Addresses

```
SRAM write:    0x33500000 | (phys_addr << 4)    # GBCore SRAM[phys_addr]
GBCore config: 0x33400010                        # Memory region base addresses
Attn config:   0x330D0010                        # AttentionConfig register
Attn perf:     0x330D0020                        # Performance counters (read)
Attn start:    0x33000030                        # Trigger attention
```

## AXI Command Sequence for Attention (N=16)

### SRAM Layout
- Q: addresses 0-15
- K: addresses 16-31
- V: addresses 32-47
- O: addresses 48-63 (output)

### Step-by-step

1. **Write Q[0..15] to SRAM**: 16 writes to 0x33500000, 0x33500010, ..., 0x335000F0
2. **Write K[0..15] to SRAM**: 16 writes to 0x33500100, 0x33500110, ..., 0x335001F0
3. **Write V[0..15] to SRAM**: 16 writes to 0x33500200, 0x33500210, ..., 0x335002F0
4. **Configure GBCore**: Write to 0x33400010
   - 128-bit data: `{base3<<16|1, base2<<16|1, base1<<16|1, base0<<16|1}`
   - base0=0 (Q), base1=16, base2=32, base3=48
5. **Configure Attention**: Write to 0x330D0010
   - Bit layout (128-bit):
     - [0] is_valid=1
     - [10:8] q_memory_index=0
     - [18:16] k_memory_index=1
     - [26:24] v_memory_index=2
     - [34:32] o_memory_index=3
     - [55:48] seq_len=16
     - [71:64] head_dim=16
     - [87:80] num_tiles=1
6. **Start**: Write to 0x33000030 (data=0)
7. **Poll interrupt**: Read 0x570 until stable and >10
8. **Read output**: 16 reads from SRAM addresses 48-63 (0x33500300, ..., 0x335003F0)
9. **Read perf counters**: Read from 0x330D0020
   - [31:0] cycle_count, [63:32] gb_read_count, [95:64] gb_write_count

## Host Driver Reference

Key functions in `design_top.c`:
- `attn_write_sram(bar_handle, phys_addr, data[4])` — write 128-bit to GBCore SRAM
- `attn_read_sram(bar_handle, phys_addr, data_out[4])` — read 128-bit from GBCore SRAM (raw AXI, no expected check)
- `attn_configure_gbcore(bar_handle, base0, base1, base2, base3)` — set 4 memory region bases
- `attn_configure(bar_handle, seq_len, num_tiles)` — configure attention with fixed q=0,k=1,v=2,o=3
- `attn_start(bar_handle)` — trigger computation
- `attn_wait_done(bar_handle, &cycles)` — poll interrupt counter until stable
- `attn_read_perf(bar_handle, &cycles, &reads, &writes)` — read performance counters
- `attn_fpga_test(bar_handle, 16)` — full end-to-end test for N=16

## R-Channel Data Unpacking

Read data from R channel is 141 bits packed in 5×32-bit words. User data is in bits [137:10]:
```c
data[0] = (word[0] >> 10) | ((word[1] & 0x3FF) << 22);
data[1] = (word[1] >> 10) | ((word[2] & 0x3FF) << 22);
data[2] = (word[2] >> 10) | ((word[3] & 0x3FF) << 22);
data[3] = (word[3] >> 10) | ((word[4] & 0x3FF) << 22);
```

## SV Testbench Modification (if needed for hw_sim attention test)

To add an attention-specific hw_sim test, create `design_top/verif/tests/design_top_attn_test.sv` following the same pattern as `design_top_base_test.sv`, and add to `Makefile.tests`:
```
design_top_attn_test: TEST=design_top_attn_test
design_top_attn_test: all
```

Then run: `make hw_sim` after changing the Makefile target.

## Expected Results

| Variant | GB Reads (N=64) | GB Writes (N=64) | Sim Clocks (N=64) |
|---------|-----------------|-------------------|--------------------|
| Unfused | ~576 | ~576 | 84,296 |
| Partial Fused | ~320 | ~320 | 131,541 |
| Fully Fused | ~640 | ~64 | 66,454 |

FPGA wall-clock @ 125 MHz = sim_clocks × 8ns

## Known Issues / Gotchas

1. **setup.sh has a bug in generate_afi**: It sets `CL_DESIGN_NAME=counter` instead of `design_top`. May need to fix before running generate_afi.
2. **concat_Top.v has `// synopsys translate` lines**: `make fpga_build` strips them automatically via sed.
3. **Runtime Makefile builds from `../src/design_top.c`**: Make sure design_top.c has `#include <string.h>` for memset/memcpy and `<unistd.h>` for usleep.
4. **Interrupt counter**: Counts at 125 MHz. `interrupt_cycles * 8ns` = wall-clock time.
5. **run_fpga_test target**: Currently runs `sudo ./design_top 0` (base test). For attention, run manually: `sudo ./design_top 0 attn`.
