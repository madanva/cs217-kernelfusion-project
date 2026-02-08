`include "common_base_test.svh"
`include "design_top_defines.vh"

module design_top_base_test();
import tb_type_defines_pkg::*;

  // File to store AXI commands
  string axi_commands_file = "src/Top/axi_commands_test.csv";
  
  // Structure to hold a single AXI command
  typedef struct {
    string op;
    logic [49:0] addr;
    logic [144:0] data;
  } AXICommand;

  // Dynamic array to store the commands
  AXICommand axi_commands[$];

  // Task to read and parse the CSV file
  task load_axi_commands();
    int file;
    string line;
    string op;
    int addr;
    string data_str;
    int res;
    string dummy;


    file = $fopen(axi_commands_file, "r");
    if (file == 0) begin
      $error("Could not open axi_commands_test.csv");
      $finish;
    end

    while ($fgets(line, file)) begin
        // Replace commas with spaces for easier parsing
        for (int i = 0; i < line.len(); i++) begin
            if (line[i] == ",") begin
                line[i] = " ";
            end
        end
        
        // Use sscanf to parse the line
        res = $sscanf(line, "%s %s %h %s %s", dummy, op, addr, data_str, dummy);

        if (res == 5) begin
            AXICommand cmd;
            cmd.op = op;
            cmd.addr = addr;
            cmd.data = data_str.atohex();
            axi_commands.push_back(cmd);
        end
    end
    $fclose(file);
  endtask

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

  task automatic top_read(input logic [49:0] addr, output logic [140:0] data);
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
    logic [159:0] read_data;
    for (int i = 0; i < LOOP_TOP_AXI_R; i++) begin
        logic [31:0] temp_data;
        ocl_rd32(ADDR_TOP_AXI_R_START + i*4, temp_data);
        read_data[i*32 +: 32] = temp_data;
    end
    data = read_data[140:0];
  endtask

  // =========================================================================
  // Main Test Sequence
  // =========================================================================
  initial begin
    // Power up the testbench
    tb.power_up(.clk_recipe_a(ClockRecipe::A0),
                .clk_recipe_b(ClockRecipe::B0),
                .clk_recipe_c(ClockRecipe::C0));

    #500ns;

    // Load AXI commands from file
    load_axi_commands();

    // Execute AXI commands
    foreach (axi_commands[i]) begin
      if (axi_commands[i].op == "W") begin
        $display("Writing to address %h", axi_commands[i].addr);
        top_write(axi_commands[i].addr, axi_commands[i].data);
      end else if (axi_commands[i].op == "R") begin
        logic [140:0] read_data;
        $display("Reading from address %h", axi_commands[i].addr);
        top_read(axi_commands[i].addr, read_data);
        // Optional: Compare read_data with expected_data from CSV
        if (read_data !== axi_commands[i].data[140:0]) begin
            $error("Read data mismatch at address %h: expected %h, got %h", axi_commands[i].addr, axi_commands[i].data[140:0], read_data);
        end
      end
    end

    #500ns;
    tb.power_down();
    
    $display("---- TEST FINISHED SUCCESSFULLY ----");
    
    $finish;
  end
endmodule