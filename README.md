# CS 217 Project: Kernel Fusion for Attention on FPGA

## Overview

FPGA-based attention accelerator comparing three fusion variants of scaled dot-product attention:

**Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V**

| Variant | Fusion Level | Description |
|---------|-------------|-------------|
| **Unfused** | None | 3 separate phases: QK^T → Softmax (NMP) → V multiply. All intermediates written to SRAM. |
| **Partial Fused** | QK^T + Softmax | Online softmax fused with QK^T computation. Attention weights still materialized. |
| **Fully Fused** | QK^T + Softmax + V | FlashAttention-style single pass. All intermediates in registers, 2-thread dataflow. |

Parameters: N=16/64, d=16, tile_size=16, int8 I/O, 32-bit fixed-point intermediates.

Built on the Stanford CS 217 lab infrastructure (GBCore SRAM, NMP, MatchLib Connections, Catapult HLS).

## Results

### HLS Synthesis (CLK_PERIOD=4.0ns)

|  | Unfused | Partial Fused | Fully Fused |
|--|---------|---------------|-------------|
| Throughput (II) | 3 | 3 | MemCtrl=1, Compute=3 |
| DSPs | 96 | 95 | 186 |
| LUTs | 8,667 | 16,650 | 17,817 |
| Total Area Score | 72,699 | 80,015 | 141,879 |

### RTL Scverify (N=64)

|  | Unfused | Partial Fused | Fully Fused |
|--|---------|---------------|-------------|
| Sim Clocks | 84,296 | 130,793 | 66,454 |
| GB Reads | 8,768 | 8,512 | 8,256 |
| GB Writes | 576 | 320 | 64 |
| Write Reduction | 1x | 1.8x | 9x |
| Speedup | 1x | 0.64x | 1.27x |

All 3 variants: TESTBENCH PASS (SystemC + HLS RTL scverify).

## File Structure

```
src/
  include/
    AttentionSpec.h          # Attention config, counters, types
    AttnDatapath.h           # Dot product, weighted accum helpers
  Top/GBPartition/GBModule/
    AttnUnfused/             # Unfused attention + testbench
    AttnPartialFused/        # Partial fused attention + testbench
    AttnFullyFused/          # Fully fused attention + testbench
    GBCore/GBCore.h          # SRAM controller (attention ports added)
    GBModule.h               # Top-level wiring (AXI region 0xD)
hls/
  Top/GBPartition/GBModule/
    AttnUnfused/Makefile     # HLS synthesis Makefiles
    AttnPartialFused/Makefile
    AttnFullyFused/Makefile
reports/
  attention_results.txt      # Full results summary
  scverify/                  # Saved simulation logs
design_top/                  # AWS F2 FPGA build environment
```

## Building

### SystemC Simulation (standalone testbenches)
```bash
# Inside rhel8 container
cd src/Top/GBPartition/GBModule/AttnUnfused
make clean && make sim_test && ./sim_test
```

### HLS Synthesis + RTL Scverify
```bash
# Inside rhel8 container
cd hls/Top/GBPartition/GBModule/AttnUnfused
make clean && make hls
```

### Top-level
```bash
python3 test.py --action systemc_sim   # SystemC sim
python3 test.py --action rtl_sim       # HLS + RTL scverify
```

## AXI Address Map (Attention)

| Address | Function |
|---------|----------|
| `0x330D0010` | Attention config register |
| `0x33000030` | Attention start trigger |

Attention uses AXI tag `0xD` for configuration and start trigger `local_index=0x3`.

## Key Design Decisions

- **Online softmax** (partial + fully fused): Running max/sum avoids two-pass softmax, enables streaming
- **addtree pragma**: Fixes SCHD-3 feedback path on running_sum accumulation (O(log N) vs O(N) depth)
- **2-thread dataflow** (fully fused): Separates MemCtrl (II=1) from Compute (II=3) via ac_channel
- **Performance metric**: Scverify simulation timestamps (sc_time_stamp) as ground truth, matching lab infrastructure approach
