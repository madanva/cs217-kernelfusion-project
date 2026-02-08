`include "common_base_test.svh"
`include "design_top_defines.vh"

module design_top_base_test();
import tb_type_defines_pkg::*;

  typedef struct {
    logic [49:0] addr;
    logic [144:0] data;
  } AxiWriteCommand;

  typedef struct {
    logic [49:0] addr;
    logic [140:0] data;
  } AxiReadCommand;


  task automatic ocl_wr32(input logic [ADDR_WIDTH_OCL - 1 : 0] addr, input logic [WIDTH_AXI - 1:0] data);
    tb.poke_ocl(.addr(addr), .data(data));
  endtask

  task automatic ocl_rd32(input logic [ADDR_WIDTH_OCL - 1 : 0] addr, output logic [WIDTH_AXI - 1:0] data);
    tb.peek_ocl(.addr(addr), .data(data));
  endtask

  task automatic top_write(input logic [49:0] addr, input logic [144:0] data);
    // Write address to AW channel
    for (int i = 0; i < LOOP_TOP_AXI_AW; i++) begin
        logic [31:0] temp_addr;
        temp_addr = addr[i*32 +: 32];
        if (i == LOOP_TOP_AXI_AW - 1) begin
          temp_addr = {18'd0, addr[49:32]};
        end
        ocl_rd32(ADDR_TOP_AXI_AW_START + i*4, temp_addr);

    end
        
   // Write data to W channel
    for (int i = 0; i < LOOP_TOP_AXI_W; i++) begin
        logic [31:0] temp_data;
        temp_data = data[i*32 +: 32];
        if (i == LOOP_TOP_AXI_W - 1) begin
          temp_data = {17'd0, data[144:128]};
        end
        ocl_wr32(ADDR_TOP_AXI_W_START + i*4, temp_data);
    end
  endtask

  task automatic top_read(input logic [49:0] addr, output logic [140:0] data, input logic [144:0] expected_read_data);
    logic [159:0] read_data;

    // Write address to AR channel
    for (int i = 0; i < LOOP_TOP_AXI_AR; i++) begin
        logic [31:0] temp_addr;
        temp_addr = addr[i*32 +: 32];
        if (i == LOOP_TOP_AXI_AR - 1) begin
          temp_addr = {18'd0, addr[49:32]};
        end
        ocl_rd32(ADDR_TOP_AXI_AR_START + i*4, temp_addr);
    end

    // Read data from R channel
    for (int i = 0; i < LOOP_TOP_AXI_R; i++) begin
        logic [31:0] temp_data;
        ocl_rd32(ADDR_TOP_AXI_R_START + i*4, temp_data);
        read_data[i*32 +: 32] = temp_data;
    end
    data = read_data[140:0];

    if (read_data[127:0] != expected_read_data[127:0])
      $display(" Read data vs expected data mismatch! Read data = %h, Expected data = %h", read_data[127:0], expected_read_data[127:0]);

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
    write_command.addr = 0;
    write_command.data = 128'b0000000000000000;
    read_command.addr = 0;
    read_command.data = 128'b0000000000000000;


    write_command.addr = 32'h33500000;
    write_command.data = 128'hD4C04352A0A882BF584169B29EE3E635;

    read_command.addr = 32'h33500000;

    top_write(write_command.addr, write_command.data);
    top_read(read_command.addr, read_command.data, write_command.data);


    #500ns;

    
    #500ns;
    tb.power_down();
    
    $display("---- TEST FINISHED SUCCESSFULLY ----");
    
    $finish;
  end
endmodule