`timescale 1ns/1ps
`define CYCLE 12
`define END_CYCLES 500

module tb_layernorm();
// ===== System Signals =====
    reg clk;
    reg [255:0] in_data;
    wire [255:0] out_data;
    reg data_in_valid;
    reg data_out_ready;
    wire data_out_valid;
    wire data_in_ready;
    reg rst;
    reg start_count;
    integer i;
    reg [255:0]weights;
    reg [255:0]bias;
    reg [31:0] in_scale;
    reg [31:0] weight_scale;
    reg [31:0] bias_scale;
    reg [31:0] out_scale;
    reg [10:0] temp;
    real in_data_real[127:0];
    real out_data_real[127:0];
    real weights_real[127:0];
    real bias_real[127:0];
    integer fp_r, fp_w, cnt;
    integer count;
    integer cycle_count;
     // ===== Module instantiation =====
    layernorm layernorm_inst(
        .clk(clk),
        .rst(rst),
        .data_in_valid(data_in_valid),
        .data_out_ready(data_out_ready),
        .in_data(in_data),
        .weights(weights),
        .bias(bias),
        .in_scale(1995),
        .weight_scale(939),
        .bias_scale(1045),
        .out_scale(1499),
        .data_out_valid(data_out_valid),
        .data_in_ready(data_in_ready),
        .out_data(out_data)
    );
    // ===== System reset ===== //
    initial begin
        clk = 0;
        in_data = 0;
        rst = 1;
        start_count = 0;
    end

    // ===== Clk fliping ===== //
    always #(`CYCLE/2) begin
        clk = ~clk;
    end 

    // ===== Set simulation info ===== //
    initial begin
        $dumpfile("layernorm.vcd");
        $dumpvars("+all");
    end

    // ===== Simulating  ===== //
    initial begin
        //stream input data into softmax, there are seq_len data
        fp_r = $fopen("layernorm_0_weights.txt", "r");
        count = 0;
        while(!$feof(fp_r)) begin
            cnt = $fscanf(fp_r, "%f", weights_real[count]);
            count = count + 1;
        end
        $fclose(fp_r);
        count = 0;

        fp_r = $fopen("layernorm_0_bias.txt", "r");
        while(!$feof(fp_r)) begin
            cnt = $fscanf(fp_r, "%f", bias_real[count]);
            count = count + 1;
        end
        $fclose(fp_r);

        count = 0;
        fp_r = $fopen("layernorm_0_input.txt", "r");
        while(!$feof(fp_r)) begin
            cnt = $fscanf(fp_r, "%f", in_data_real[count]);
            count = count + 1;
        end
        $fclose(fp_r);

        for(i=0; i<32; i=i+1)
            in_data[i*8+7 -: 8] = in_data_real[i];
        
        for(i=0; i<32; i=i+1)
            weights[i*8+7 -: 8] = weights_real[i];
        
        for(i=0; i<32; i=i+1)
            bias[i*8+7 -: 8] = bias_real[i];
            
        $display("Reset System");
        @(negedge clk);
        rst = 1'b0;
        @(negedge clk);
        @(negedge clk);
        @(negedge clk);
        @(negedge clk);
        rst = 1'b1;
        data_in_valid = 1;
        $display("Compute start");
        @(negedge clk);
         for(i=0; i<32; i=i+1)
            in_data[i*8+7 -: 8] = in_data_real[i+32];
        
        for(i=0; i<32; i=i+1)
            weights[i*8+7 -: 8] = weights_real[i+32];
        
        for(i=0; i<32; i=i+1)
            bias[i*8+7 -: 8] = bias_real[i+32];

        @(negedge clk);
         for(i=0; i<32; i=i+1)
            in_data[i*8+7 -: 8] = in_data_real[i+64];
        
        for(i=0; i<32; i=i+1)
            weights[i*8+7 -: 8] = weights_real[i+64];
        
        for(i=0; i<32; i=i+1)
            bias[i*8+7 -: 8] = bias_real[i+64];

        @(negedge clk);
        for(i=0; i<32; i=i+1)
            in_data[i*8+7 -: 8] = in_data_real[i+96];
        
        for(i=0; i<32; i=i+1)
            weights[i*8+7 -: 8] = weights_real[i+96];
        
        for(i=0; i<32; i=i+1)
            bias[i*8+7 -: 8] = bias_real[i+96];
        @(negedge clk);
        data_in_valid = 0;
        data_out_ready = 1;
        wait(data_out_valid == 1);
        @(negedge clk);
        for(i=0 ; i<32 ; i=i+1)
            $display("out_data[%f] = %f", i, $signed(out_data[8*i+7 -: 8]));
        @(negedge clk);
        for(i=0 ; i<32 ; i=i+1)
            $display("out_data[%f] = %f", i + 32, $signed(out_data[8*i+7 -: 8]));
        @(negedge clk);
        for(i=0 ; i<32 ; i=i+1)
            $display("out_data[%f] = %f", i + 64, $signed(out_data[8*i+7 -: 8]));
        @(negedge clk);
        for(i=0 ; i<32 ; i=i+1)
            $display("out_data[%f] = %f", i + 96, $signed(out_data[8*i+7 -: 8]));
        @(negedge clk);

        $display("Simulation finish");
        $finish;
    end



endmodule

