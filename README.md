# Lab 4: System Integration and Interconnect

## Table of Contents
- [1. Introduction and Objectives](#1-introduction-and-objectives)
  - [File Structure](#file-structure)
- [2. Architecture Overview](#2-architecture-overview)
- [3. SystemC Design and Integration Tasks](#3-systemc-design-and-integration-tasks)
  - [Task 1: PE Module and Partition Integration](#task-1-pe-module-and-partition-integration)
  - [Task 2: GB Module and Partition Integration](#task-2-gb-module-and-partition-integration)
  - [Task 3: Top-Level Integration](#task-3-top-level-integration)
  - [Task 4: SystemC Simulation and Verification](#task-4-systemc-simulation-and-verification)
- [4. HLS to RTL and FPGA Implementation](#4-hls-to-rtl-and-fpga-implementation)
  - [4.1. High-Level Synthesis (HLS)](#41-high-level-synthesis-hls)
  - [4.2. RTL Simulation and FPGA Build (on AWS F2)](#42-rtl-simulation-and-fpga-build-on-aws-f2)
- [5. Submission](#5-submission)

## 1. Introduction and Objectives

This lab focuses on the integration of the various hardware modules developed in previous labs to build the complete accelerator system. The primary objective is to implement the interconnect between the Global Buffer (GB) and Processing Element (PE) partitions, enabling end-to-end data and control flow.

By the end of this lab, you will have:
*   A deep understanding of the hierarchical SystemC design of the accelerator.
*   Implemented the port connections and wiring for all major modules.
*   Wired up the AXI infrastructure, including the AXI-to-RVA converters and the AXI Splitter, to enable external control.
*   Verified the functionality of the fully integrated design through simulation.

### File Structure

The key files for this lab are located in the `src` directory.

-   `src/Top/`: The top-level module where you will connect the GB and PE partitions.
-   `src/DataBus/`: Contains the modules for handling data and control signal distribution between the GB and PEs.
-   `src/GBPartition/`: Contains the `GBPartition` and its internal `GBModule`.
-   `src/PEPartition/`: Contains the `PEPartition` and its internal `PEModule`.
-   `src/include/`: Contains shared header files, including `AxiSpec.h` and `SM6Spec.h`.

## 2. Architecture Overview

The top-level design consists of three main components:

-   **GBPartition**: The Global Buffer partition, which is responsible for storing weights and activations. It contains the `GBModule`.
-   **PEPartition**: The Processing Element partition. The design instantiates multiple PEs. Each partition contains a `PEModule`.
-   **AxiSplitter**: An AXI4 interconnect that routes AXI transactions from the host to the appropriate partition (GB or one of the PEs) based on the address.
-   **DataBus**: A set of modules that manage the broadcasting of data from the GB to all PEs (`GBSend`), the collection of results from PEs back to the GB (`GBRecv`), and the distribution/aggregation of control signals (`PEStart`, `PEDone`).

## 3. SystemC Design and Integration Tasks

This section details the `TODO` items that need to be completed. The tasks involve connecting ports between modules at various levels of the design hierarchy. We recommend a bottom-up approach, starting with the innermost modules.

### Task 1: PE Module and Partition Integration
-   **`src/PEPartition/PEModule/PEModule.h`**
    1.  In `PERVA`, arbitrate between the two RVA output channels (`pe_rva_out` and `act_rva_out`).
    2.  Test the implementation with `cd src/PEPartition/PEModule; make`
-   **`src/PEPartition/PEPartition.h`**
    1.  Define the external ports for the `PEPartition` module (`TODO #1`).
    2.  Connect the ports of the `rva_inst` and `pemodule_inst` instances (`TODO #2`).
    3.  Test the implementation with `cd src/PEPartition; make`

### Task 2: GB Module and Partition Integration
-   **`src/GBPartition/GBModule/GBModule.h`**
    1.  In `GBRVA`, decode the AXI address to send a start signal to the correct sub-module (`TODO #1`).
    2.  In `GBDone`, aggregate `done` signals from all sub-modules (`TODO #2`).
    3.  Test the implementation with `cd src/GBPartition/GBModule; make`
-   **`src/GBPartition/GBPartition.h`**
    1.  Define ports, instantiate `GBModule` and the AXI-to-RVA converter, and connect them (`TODO #1`).
    2.  Test the implementation with `cd src/GBPartition; make`

### Task 3: Top-Level Integration
-   **`src/Top/Top.h`**
    1.  Connect the `GBPartition` instance (`TODO #1`).
    2.  Instantiate and connect the `PEPartition` modules (`TODO #2`).
    3.  Connect the `AxiSplitter` instance (`TODO #3`).
    4.  Connect the databus and interrupt handling modules (`TODO #4`).
    5.  **Note:** The top-level simulation is driven by AXI commands from `src/Top/axi_commands_test.csv`. You will need to create this file to test your implementation. Please use the `axi_commands_test.csv` files located in the `src/PEPartition` and `src/GBPartition` directories as a reference for the command format. Describe the test you are trying to perform in the writeup. Please note that Top testbench is designed to pass only on an interrupt, and an interrupt is only generated by the GB.
    6.  Test the final integration with `cd src/Top; make`

## 4. HLS to RTL and FPGA Implementation

This section outlines the workflow for generating RTL from your SystemC design and implementing it on the AWS F2 FPGA.

### 4.1. High-Level Synthesis (HLS)

To manage complexity and avoid the high runtime of a flat, top-level synthesis, we will follow a hierarchical, bottom-up approach to generate the RTL.

1.  **Synthesize `PEModule` and `GBModule`:** These can be synthesized in parallel.
    ```bash
    # Synthesize PEModule
    cd hls/PEPartition/PEModule
    make clean && make
    ```
    ```bash
    # Synthesize GBModule
    cd hls/GBPartition/GBModule
    make clean && make
    ```

2.  **Synthesize `PEPartition` and `GBPartition`:** Once the inner modules are complete, synthesize the partition-level modules.
    ```bash
    # Synthesize PEPartition
    cd hls/PEPartition
    make clean && make
    ```
    ```bash
    # Synthesize GBPartition
    cd hls/GBPartition
    make clean && make
    ```

3.  **Synthesize the Top-Level Module:** Finally, run HLS for the top-level design.
    ```bash
    cd hls/Top
    cp ../src/Top/axi_commands_test.csv .
    make clean && make
    ```

## 4. Submission

-   Generate your submission zip file using:
    ```bash
    make submission
    ```
- Make sure you have the following files
  - `src/PEPartition/PEModule/PEModule.h`
  - `src/PEPartition/PEPartition.h`
  - `src/GBPartition/GBModule/GBModule.h`
  - `src/GBPartition/GBPartition.h`
  - `src/Top/Top.h`
  - `src/Top/axi_commands_test.csv`

-   Follow the course guidelines to submit the generated `lab4-submission.zip` and any required writeup.
-   Ensure your writeup includes the simulation and implementation log files as requested in the lab handout.
