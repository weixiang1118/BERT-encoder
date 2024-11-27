`timescale 1ns/1ps
`define CYCLE 12
`define END_CYCLES 500

module tb_gelu();
// ===== System Signals =====
    reg clk;
    reg [255:0] in_data;
    wire [255:0] out_data;
    reg start_count;
    integer i;
    reg rst;
    wire data_in_ready;
    reg data_in_valid;
    reg data_out_ready;
    wire data_out_valid;


     // ===== Module instantiation =====
    GELU gelu_inst(
        .clk(clk),
        .rst(rst),
        .data_in_valid(data_in_valid),
        .data_out_ready(data_out_ready),
        .in_data(in_data),
        .in_scale(2005),
        .out_scale(1072),
        .data_out_valid(data_out_valid),
        .data_in_ready(data_in_ready),
        .out_data(out_data)
    );
    // ===== System reset ===== //
    initial begin
        clk = 0;
        in_data = 0;
        start_count = 0;
    end

    // ===== Cycle count ===== //
    initial begin
        wait(start_count == 1);
        start_count = 0;
    end

    // ===== Clk fliping ===== //
    always #(`CYCLE/2) begin
        clk = ~clk;
    end 

    // ===== Set simulation info ===== //
    initial begin
        $dumpfile("GELU.vcd");
        $dumpvars("+all");
    end

    // ===== Simulating  ===== //
    initial begin
        @(negedge clk);
        in_data[7:0] = 30;
        in_data[15:8] = 25;
        in_data[23:16] = 24;
        in_data[31:24] = -7;
        in_data[39:32] = 51;
        data_in_valid = 1;
        $display("Compute start");
        @(negedge clk);
        start_count = 1;
        @(negedge clk);
        start_count = 0;
        @(negedge clk);
        data_out_ready = 1;
        for(i=0 ; i<32 ; i=i+1)
            $display("out_data[%f] = %f", i, $signed(out_data[8*i+7 -: 8]));
        $display("Simulation finish");
        $finish;
    end



endmodule

