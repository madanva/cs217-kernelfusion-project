# Lab 4: System Integration and Interconnect

## Table of Contents
- [1. Introduction](#1-introduction)
  - [File Structure](#file-structure)
- [2. Architecture Overview](#2-architecture-overview)
- [3. SystemC test and HLS to RTL](#3-systemc-test-and-hls-to-rtl)
- [4. FPGA Implementation](#4-fpga-implementation)
- [5. How was Timing fixed?](#5-how-was-timing-fixed)
- [6. Documentation](#6-documentation)


## 1. Introduction 
This repository contains the solution to Lab 4 with a timing-clean AWS environment. 

### File Structure

The repository is organized into the following directories:

-   `src/`: Contains the SystemC source code for the accelerator. This is where you will spend most of your time for this lab.
    -   `src/include/`: Shared header files, including definitions for AXI signals (`AxiSpec.h`) and other specifications.
    -   `src/DataBus/`: Modules for handling data and control signal distribution between the GB and PEs.
    -   `src/Top/GBPartition/`: The Global Buffer partition, which includes the `GBModule` and its sub-modules.
    -   `src/Top/PEPartition/`: The Processing Element partition, which includes the `PEModule` and its sub-modules.
    -   `src/Top/`: The top-level module that integrates the GB and PE partitions.
-   `hls/`: Contains the scripts and Makefiles for running the High-Level Synthesis (HLS) flow. The structure mirrors the `src` directory.
-   `design_top/`: Contains the files for building the design for the AWS F2 FPGA, including Verilog/SystemVerilog files, constraints, and build scripts.
    -   `design_top/design/`: Synthesized RTL files and other design sources.
    -   `design_top/verif/`: Verification environment for the RTL design.
    -   `design_top/build/`: Scripts and constraints for synthesis and implementation.
-   `scripts/`: Utility scripts for the project.
    -   `scripts/hls/`: Scripts for the HLS flow.
    -   `scripts/aws/`: Scripts for interacting with AWS.
-   `docs/`: Contains documentation for the project, including setup guides.
-   `reports/`: Directory for storing reports from HLS and FPGA builds.

## 2. Architecture Overview

The top-level design consists of three main components:

-   **GBPartition**: The Global Buffer partition, which is responsible for storing weights and activations. It contains the `GBModule`.
-   **PEPartition**: The Processing Element partition. The design instantiates multiple PEs. Each partition contains a `PEModule`.
-   **AxiSplitter**: An AXI4 interconnect that routes AXI transactions from the host to the appropriate partition (GB or one of the PEs) based on the address.
-   **DataBus**: A set of modules that manage the broadcasting of data from the GB to all PEs (`GBSend`), the collection of results from PEs back to the GB (`GBRecv`), and the distribution/aggregation of control signals (`PEStart`, `PEDone`).

## 3. SystemC test and HLS to RTL
1. SystemC test - `python3 test.py --action systemc_sim`
2. RTL generation and sim - `python3 test.py --action rtl_sim`

## 4. FPGA Implementation

**Disclaimer:** If your `hw_sim` fails with a testbench error, please check out the following commit for the `aws-fpga` repo before sourcing `hdk_setup.sh`: `git checkout 2f1f343259dcb794adc596a76c31593280fd7c7b`

Clone your lab repository to AWS F2 and set up the environment. This should be similar to previous labs:

```bash
# SSH into AWS F2 instance

# Source AWS F2 environment
cd ~/aws-fpga
source hdk_setup.sh
source sdk_setup.sh

# Move to design_top folder
cd [path-to-lab4-repo]/design_top
source setup.sh
```

Run hardware simulation on AWS F2 for the synthesized RTL:

```bash
# Run RTL simulation
make hw_sim
```

Build the FPGA bitstream, generate the AFI, and program the FPGA:

```bash
cd design_top

make fpga_build          # should take about 2.5 hours
make generate_afi

# Wait for AFI to become available
make check_afi_available

# Once available
make program_fpga
make run_fpga_test
```
### 5. How was Timing fixed?

Credits to @jadbitar for the timing-fixed solution code.

1. HLS closed at 250MHz to prevent long buffer paths in RTL when closed at 125MHz
2. The AWS clock `clk_main_a0` is put through a clock divider to supply 125MHz to all the components. Appropriate synchronization primitives were used.

Files changed from Lab 4 release repo
1. `design_top/design/design_top.sv`
2. `design_top/build/constraints/cl_timing_user.xdc`
3. `design_top/build/scripts/synth_design_top.tcl`

## 6. Documentation

Refer to the following documentation for SystemC and MatchLib, found on both Canvas and in `/cad/mentor/2024.2_1/Mgc_home/shared/pdfdocs`:

- `ac_datatypes_ref.pdf`: Algorithmic C datatypes reference manual. Specifically, see `ac_fixed` and `ac_float` classes.
- `ac_math_ref.pdf`: Algorithmic C math library reference manual. Specifically, see functions for power, reciprocal, and square root.
- `connections-guide.pdf`: Documentation for the Connections library, including detailed information on `Push`/`Pop` semantics and coding best practices.
- `catapult_useref.pdf`: User reference manual for Catapult HLS, including pragmas and synthesis directives. 
- `https://nvlabs.github.io/matchlib`: MatchLib online documentation, including component reference and tutorials.