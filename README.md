# Lab 4: System Integration and Interconnect

## Table of Contents
- [1. Introduction and Objectives](#1-introduction-and-objectives)
  - [File Structure](#file-structure)
- [2. Architecture Overview](#2-architecture-overview)
- [3. SystemC Design and Integration Tasks](#3-systemc-design-and-integration-tasks)
  - [Task 1: PE Module and Partition Integration](#task-1-pe-module-and-partition-integration)
  - [Task 2: GB Module and Partition Integration](#task-2-gb-module-and-partition-integration)
  - [Task 3: Top-Level Integration](#task-3-top-level-integration)
  - [Task 4: SystemC Sanity Testing](#task-4-systemc-sanity-testing)
- [4. HLS to RTL and FPGA Implementation](#4-hls-to-rtl-and-fpga-implementation)
  - [4.1. High-Level Synthesis (HLS)](#41-high-level-synthesis-hls)
  - [4.2. RTL to FPGA Implementation](#42-rtl-to-fpga-implementation)
  - [4.3. Analyzing Timing Failures](#43-analyzing-timing-failures)
- [5. Submission](#5-submission)
  - [5.1. Submission Write-up](#51-submission-write-up)
  - [5.2. Code Submission](#52-code-submission)
- [6. Documentation](#6-documentation)


## 1. Introduction and Objectives

This lab focuses on the integration of the various hardware modules developed in previous labs to build the complete accelerator system. The primary objective is to implement the interconnect between the Global Buffer (GB) and Processing Element (PE) partitions, enabling end-to-end data and control flow.

By the end of this lab, you will have:
*   A deep understanding of the hierarchical SystemC design of the accelerator.
*   Implemented the port connections and wiring for all major modules.
*   Wired up the AXI infrastructure, including the AXI-to-RVA converters and the AXI Splitter, to enable external control.
*   Verified the functionality of the fully integrated design through simulation.

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

## 3. SystemC Design and Integration Tasks

This section details the `TODO` items that need to be completed. The tasks involve connecting ports between modules at various levels of the design hierarchy. We recommend a bottom-up approach, starting with the innermost modules.

### Task 1: PE Module and Partition Integration
-   **`src/Top/PEPartition/PEModule/PEModule.h`**: Complete the `TODO`s in the file and verify with `cd src/Top/PEPartition/PEModule; make`.
-   **`src/Top/PEPartition/PEPartition.h`**: Complete the `TODO`s in the file and verify with `cd src/Top/PEPartition; make`.

### Task 2: GB Module and Partition Integration
-   **`src/Top/GBPartition/GBModule/GBModule.h`**: Complete the `TODO`s in the file and verify with `cd src/Top/GBPartition/GBModule; make`.
-   **`src/Top/GBPartition/GBPartition.h`**: Complete the `TODO`s in the file and verify with `cd src/Top/GBPartition; make`.

### Task 3: Top-Level Integration
-   **`src/Top/Top.h`**
    1.  Connect the `GBPartition` instance (`TODO #1`).
    2.  Instantiate and connect the `PEPartition` modules (`TODO #2`).
    3.  Connect the `AxiSplitter` instance (`TODO #3`).
    4.  Connect the databus and interrupt handling modules (`TODO #4`).
    5.  **Note:** The top-level simulation is driven by AXI commands from `src/Top/axi_commands_test.csv`. A test is provided. In your write-up, describe the test's functionality, including how the PE receives inputs and weights, how the GB stores the PE output, and how an NMP operation is performed (if applicable). Also note that whenever 0xDEADBEEF is written to an address location, consider it as a NOP instruction.
    6.  Test the final integration with `cd src/Top; make`

### Task 4: SystemC Sanity Testing
Once your implementation is complete, trigger a full sanity test from the repository's top folder by running: `python3 test.py --action systemc_sim`. This will generate a log in the `reports` directory for your submission.

## 4. HLS to RTL and FPGA Implementation

This section outlines the workflow for generating RTL from your SystemC design and implementing it on the AWS F2 FPGA.

### 4.1. High-Level Synthesis (HLS)

To manage complexity and avoid the high runtime of a flat, top-level synthesis, we will follow a hierarchical, bottom-up approach to generate the RTL. If you are confident in your design, you can directyl trigger all modules rtl generation with `python3 test.py --action hls_sim` (this should take about 2 hours).

1.  **Synthesize `PEModule` and `GBModule`:** These can be synthesized in parallel.
    ```bash
    # Synthesize PECore and ActUnit parallely

    cd hls/Top/PEPartition/PEModule/ActUnit
    make clean && make hls

    cd hls/Top/PEPartition/PEModule/PECore
    make clean && make hls

    # Synthesize PEModule
    cd hls/Top/PEPartition/PEModule/
    make clean && make hls
    ```

    ```bash
    # Synthesize NMP, GBCore and GBControl parallely

    cd hls/Top/GBPartition/GBModule/NMP
    make clean && make hls

    cd hls/Top/GBPartition/GBModule/GBCore
    make clean && make hls

    cd hls/Top/GBPartition/GBModule/
    make clean && make hls

    # Synthesize GBModule
    cd hls/GBPartition/GBModule
    make clean && make
    ```

2.  **Synthesize `PEPartition` and `GBPartition`:** Once the inner modules are complete, synthesize the partition-level modules.
    ```bash
    # Synthesize PEPartition
    cd hls/PEPartition
    make clean && make hls
    ```
    ```bash
    # Synthesize GBPartition
    cd hls/GBPartition
    make clean && make hls
    ```

3.  **Synthesize the Top-Level Module:** Finally, run HLS for the top-level design.
    ```bash
    cd hls/Top
    make clean && make hls
    ```

4. Make sure to copy the required files and reports to their respective folders. If you have used `test.py` for HLS to RTL generation, this step is automatically covered. Otherwise, please run `make copy_rtl` from the top of the repository.

### 4.2. RTL to FPGA Implementation

Clone your lab respository to AWS F2 and set up the environment. Again, this should be similar to previous labs:

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
### 4.3. Analyzing Timing Failures

You may observe that some or all AXI read operations fail to produce the expected results. If this occurs, inspect the timing reports in `design_top/build/reports/` for any violations.

In your write-up, address the following points:
*   Correlate the test failures with the specific issues identified in the timing report.
*   Propose potential solutions to resolve these timing issues. Consider options such as adjusting the clock frequency or adding new timing constraints.
*   Describe the necessary changes in the overall workflow to modify the clock frequency.
*   Explain how the clock frequency is specified in both the HLS-to-RTL and the FPGA build processes. For your reference our current clock frequency is 250MHz which is a very tight constraint for AWS F2 to achieve for our design

**Extra Credit:**
If you successfully resolve the timing violations using the methods you proposed, document the process and include it in your report for extra credit.

## 5. Submission
### Writeup Submission

### 5.1. Submission Write-up

Please submit a write-up that includes the following sections:

**1. Simulation and Test Results:**
*   A description of the top-level simulation test.
*   The log output from the SystemC simulation of the `Top` module.
*   The log output from the RTL simulation of the `Top` module (post-HLS).
*   The log output from the AWS F2 hardware simulation.
*   The log output from the AWS F2 FPGA test.

**2. Timing Failure Analysis (if applicable):**
*   An analysis of any failures observed during the AWS F2 FPGA test, correlating them with timing reports.
*   **Extra Credit:** If you resolved the timing failures, describe the steps you took to fix them and include a screenshot of the final passing FPGA test.

**3. Use of Generative AI:**
*   Document your use of any generative AI tools (e.g., ChatGPT, Claude, Copilot).
    *   What tools and workflows did you use?
    *   How did you prompt the tools? What context did you provide?
    *   What worked well and what didn't?
    *   What were the observed productivity gains?

### 5.2. Code Submission

Submit `lab4-submission.zip` generated by `make submission`. Ensure these files exist:

- Make sure you have the following files
  - `src/Top/PEPartition/PEModule/PEModule.h`
  - `src/Top/PEPartition/PEPartition.h`
  - `src/Top/GBPartition/GBModule/GBModule.h`
  - `src/Top/GBPartition/GBModule.h`
  - `src/Top/Top.h`
  - `design_top/design/concat_Top.v`
  - `design_top/design/design_top.sv`
  - `reports/aws/`
    - `fpga_test.log.txt`
    - `f2_hw_sim.log.txt`
  - `reports/hls/`
    - `rtl_sim.log.txt`
    - `systemc.log.txt`
    - `Top_hls_sim.log`
    - `Top.rpt`
  - Any file your want to include for extra credit

## 6. Documentation

Refer to the following documentation items for SystemC and MatchLib found both on Canvas and `/cad/mentor/2024.2_1/Mgc_home/shared/pdfdocs`.

- `ac_datatypes_ref.pdf`: Algorithmic C datatypes reference manual. Specifically, see `ac_fixed` and `ac_float` classes.
- `ac_math_ref.pdf`: Algorithmic C math library reference manual. Specifically, see functions for power, reciprocal, and square root.
- `connections-guide.pdf`: Documentation for the Connections library including detailed information on `Push`/`Pop` semantics and coding best practices.
- `catapult_useref.pdf`: User reference manual for Catapult HLS including pragmas and synthesis directives. 
- `https://nvlabs.github.io/matchlib`: MatchLib online documentation including component reference and tutorials.