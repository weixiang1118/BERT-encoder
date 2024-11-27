`timescale 1ns/1ps
`define CYCLE 12
`define END_CYCLES 500

module tb_softmax();
// ===== System Signals =====
    reg clk;
    reg [255:0] in_data;
    wire [255:0] out_data;
    integer i;
    reg start_count;
    reg rst;
    wire data_in_ready;
    reg data_in_valid;
    reg data_out_ready;
    wire data_out_valid;
    integer fp_r, fp_w, cnt;
    integer count;
    // ===== Module instantiation =====
    softmax softmax_inst(
        .clk(clk),
        .rst(rst),
        .data_in_valid(data_in_valid),
        .data_out_ready(data_out_ready),
        .in_data(in_data),
        .in_scale(20132),
        .out_scale(475),
        .S(28),
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

    // ===== Clk fliping ===== //
    always #(`CYCLE/2) begin
        clk = ~clk;
    end 

    // ===== Set simulation info ===== //
    initial begin
        $dumpfile("softmax.vcd");
        $dumpvars("+all");
    end

    // ===== Simulating  ===== //
    initial begin
        @(negedge clk);
        //stream input data into softmax, there are seq_len data
        fp_r = $fopen("score_0_input.txt", "r");
        count = 0;
        while(!$feof(fp_r)) begin
            cnt = $fscanf(fp_r, "%d", in_data[count*8+7 -: 8]);
            count = count + 1;
        end
        $fclose(fp_r);
        count = 0;
        
        data_in_valid = 1;
        $display("Compute start");
        @(negedge clk);
        data_out_ready = 1;
        @(negedge clk);
        for(i=0 ; i<32 ; i=i+1)
            $display("out_data[%f] = %f", i, $signed(out_data[8*i+7 -: 8]));

        $display("Simulation finish");
        $finish;
    end



endmodule

