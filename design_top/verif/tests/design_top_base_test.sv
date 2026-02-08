`include "common_base_test.svh"
`include "design_top_defines.vh"

bit test_failed = 0;

module design_top_base_test();
import tb_type_defines_pkg::*;

  typedef struct {
    logic [31:0] addr;
    logic [127:0] data;
  } AxiWriteCommand;

  typedef struct {
    logic [31:0] addr;
    logic [127:0] data;
    logic [127:0] expected_read_data;
  } AxiReadCommand;


  task automatic ocl_wr32(input logic [ADDR_WIDTH_OCL - 1 : 0] addr, input logic [WIDTH_AXI - 1:0] data);
    tb.poke_ocl(.addr(addr), .data(data));
  endtask

  task automatic ocl_rd32(input logic [ADDR_WIDTH_OCL - 1 : 0] addr, output logic [WIDTH_AXI - 1:0] data);
    tb.peek_ocl(.addr(addr), .data(data));
  endtask

  task automatic top_write(input AxiWriteCommand write_command);
    // Write address to AW channel
    // Strobe = 1ffff
    logic [49:0] transfer_addr = {8'b0, write_command.addr, 10'b0};
    logic [144:0] transfer_data = {17'h1ffff, write_command.data};

    for (int i = 0; i < LOOP_TOP_AXI_AW; i++) begin
        logic [31:0] temp_addr;
        temp_addr = transfer_addr[i*32 +: 32];
        if (i == LOOP_TOP_AXI_AW - 1) begin
          temp_addr = {14'b0, transfer_addr[49:32]};
        end
        ocl_wr32(ADDR_TOP_AXI_AW_START + i*4, temp_addr);
        #10ns;
    end

    #100ns;
        
   // Write data to W channel
    for (int i = 0; i < LOOP_TOP_AXI_W; i++) begin
        logic [31:0] temp_data;
        temp_data = transfer_data[i*32 +: 32];
        if (i == LOOP_TOP_AXI_W - 1) begin
          temp_data = {17'd0, transfer_data[144:128]};
        end
        ocl_wr32(ADDR_TOP_AXI_W_START + i*4, temp_data);
        #10ns;
    end
  endtask

  task automatic top_read(AxiReadCommand read_command);
    logic [49:0] transfer_addr = {8'b0, read_command.addr, 10'b0};
    logic [159:0] transfer_data;

    // Write address to AR channel
    for (int i = 0; i < LOOP_TOP_AXI_AR; i++) begin
        logic [31:0] temp_addr;
        temp_addr = transfer_addr[i*32 +: 32];
        if (i == LOOP_TOP_AXI_AR - 1) begin
          temp_addr = {18'd0, transfer_addr[49:32]};
        end
        ocl_wr32(ADDR_TOP_AXI_AR_START + i*4, temp_addr);
        #10ns;
    end

    #100ns;

    // Read data from R channel
    for (int i = 0; i < LOOP_TOP_AXI_R; i++) begin
        logic [31:0] temp_data;
        ocl_rd32(ADDR_TOP_AXI_R_START + i*4, temp_data);
        #10ns;
        transfer_data[i*32 +: 32] = temp_data;
    end
    read_command.data = transfer_data[137:10];

    if (read_command.data != read_command.expected_read_data) begin
      $error(" Read data vs expected data mismatch! Read data = 0x%h, Expected data = 0x%h", read_command.data, read_command.expected_read_data);
      test_failed = 1'b1;
    end
    else begin
      $display("Read value matches the expected = 0x%h at 0x%h", read_command.data, read_command.addr);
    end
    

  endtask

  // =========================================================================
  // Main Test Sequence
  // =========================================================================
  initial begin
    AxiWriteCommand write_command;
    AxiReadCommand  read_command;

    // Power up the testbench
    tb.power_up(.clk_recipe_a(ClockRecipe::A0),
                .clk_recipe_b(ClockRecipe::B0),
                .clk_recipe_c(ClockRecipe::C0));

    #500ns;
    read_command.data = 'b0000000000000000;


    write_command.addr = 32'h33500000;
    write_command.data = 127'hD4C04352A0A882BF584169B29EE3E635;

    read_command.addr = 32'h33500000;
    read_command.expected_read_data = write_command.data;


    top_write(write_command);
    top_read(read_command);

    
    #500ns;
    tb.power_down();
    
    if (!test_failed)
      $display("---- TEST FINISHED SUCCESSFULLY ----");
    else
      $display("---- TEST FAILED ----");
      
    $finish;
  end
endmodule