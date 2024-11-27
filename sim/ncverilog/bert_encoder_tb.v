`timescale 1ns/1ps
`define CYCLE 12
`define END_CYCLES 500000
module bert_encoder_tb();

    

    // ===== System Signals =====
    reg clk;
    integer i, cycle_count;
    reg start_count;


    // ===== SRAM Signals =====
    wire [15:0] sram_weight_wea0;
    wire [15:0] sram_weight_addr0;
    wire [127:0] sram_weight_wdata0;
    wire [127:0] sram_weight_rdata0;
    wire [15:0] sram_weight_wea1;
    wire [15:0] sram_weight_addr1;
    wire [127:0] sram_weight_wdata1;
    wire [127:0] sram_weight_rdata1;
    
    wire [15:0] sram_act_wea0;
    wire [15:0] sram_act_addr0;
    wire [127:0] sram_act_wdata0;
    wire [127:0] sram_act_rdata0;
    wire [15:0] sram_act_wea1;
    wire [15:0] sram_act_addr1;
    wire [127:0] sram_act_wdata1;
    wire [127:0] sram_act_rdata1;

    // ===== softmax signals =====
    wire softmax_data_in_valid;
    wire softmax_data_out_ready;
    wire [255:0] softmax_in_data;
    wire [31:0] softmax_in_scale;
    wire [31:0] softmax_out_scale;
    wire softmax_data_out_valid;
    wire softmax_data_in_ready;
    wire [255:0] softmax_out_data;

    // ===== layernorm signals =====
    wire layernorm_data_in_valid;
    wire layernorm_data_out_ready;
    wire [255:0] layernorm_in_data;
    wire [255:0] layernorm_weights;
    wire [255:0] layernorm_bias;
    wire [31:0] layernorm_in_scale;
    wire [31:0] layernorm_weight_scale;
    wire [31:0] layernorm_bias_scale;
    wire [31:0] layernorm_out_scale;
    wire layernorm_data_out_valid;
    wire layernorm_data_in_ready;
    wire [255:0] layernorm_out_data;

    // ===== GELU signals =====
    wire gelu_data_in_valid;
    wire gelu_data_out_ready;
    wire [255:0] gelu_in_data;
    wire [31:0] gelu_in_scale;
    wire [31:0] gelu_out_scale;
    wire gelu_data_out_valid;
    wire gelu_data_in_ready;
    wire [255:0] gelu_out_data;

    // ===== Golden =====
    reg [127:0] golden [0:4095];

    // ===== bert_encoder Signals =====
    reg rst_n;
    reg compute_start;
    wire compute_finish;
    reg [7:0] sequence_length;

    // ===== Module instantiation =====
    bert_encoder bert_encoder_inst(
        .clk(clk),
        .rst_n(rst_n),

        .compute_start(compute_start),
        .compute_finish(compute_finish),
        .sequence_length(sequence_length),

        // weight sram, single port
        .sram_weight_wea0(sram_weight_wea0),
        .sram_weight_addr0(sram_weight_addr0),
        .sram_weight_wdata0(sram_weight_wdata0),
        .sram_weight_rdata0(sram_weight_rdata0),
        .sram_weight_wea1(sram_weight_wea1),
        .sram_weight_addr1(sram_weight_addr1),
        .sram_weight_wdata1(sram_weight_wdata1),
        .sram_weight_rdata1(sram_weight_rdata1),

        // Output sram,dual port
        .sram_act_wea0(sram_act_wea0),
        .sram_act_addr0(sram_act_addr0),
        .sram_act_wdata0(sram_act_wdata0),
        .sram_act_rdata0(sram_act_rdata0),
        .sram_act_wea1(sram_act_wea1),
        .sram_act_addr1(sram_act_addr1),
        .sram_act_wdata1(sram_act_wdata1),
        .sram_act_rdata1(sram_act_rdata1),

        // softmax module
        .softmax_data_in_valid(softmax_data_in_valid),
        .softmax_data_out_ready(softmax_data_out_ready),
        .softmax_in_data(softmax_in_data),
        .softmax_in_scale(softmax_in_scale),
        .softmax_out_scale(softmax_out_scale),
        .softmax_data_out_valid(softmax_data_out_valid),
        .softmax_data_in_ready(softmax_data_in_ready),
        .softmax_out_data(softmax_out_data),

        // layernorm module
        .layernorm_data_in_valid(layernorm_data_in_valid),
        .layernorm_data_out_ready(layernorm_data_out_ready),
        .layernorm_in_data(layernorm_in_data),
        .layernorm_weights(layernorm_weights),
        .layernorm_bias(layernorm_bias),
        .layernorm_in_scale(layernorm_in_scale),
        .layernorm_weight_scale(layernorm_weight_scale),
        .layernorm_bias_scale(layernorm_bias_scale),
        .layernorm_out_scale(layernorm_out_scale),
        .layernorm_data_out_valid(layernorm_data_out_valid),
        .layernorm_data_in_ready(layernorm_data_in_ready),
        .layernorm_out_data(layernorm_out_data),

        // GELU module
        .gelu_data_in_valid(gelu_data_in_valid),
        .gelu_data_out_ready(gelu_data_out_ready),
        .gelu_in_data(gelu_in_data),
        .gelu_in_scale(gelu_in_scale),
        .gelu_out_scale(gelu_out_scale),
        .gelu_data_out_valid(gelu_data_out_valid),
        .gelu_data_in_ready(gelu_data_in_ready),
        .gelu_out_data(gelu_out_data)
    );

    SRAM_weight_16384x128b weight_sram( 
        .clk(clk),
        .wea0(sram_weight_wea0),
        .addr0(sram_weight_addr0),
        .wdata0(sram_weight_wdata0),
        .rdata0(sram_weight_rdata0),
        .wea1(sram_weight_wea1),
        .addr1(sram_weight_addr1),
        .wdata1(sram_weight_wdata1),
        .rdata1(sram_weight_rdata1)
    );
    
    SRAM_activation_4096x128b act_sram( 
        .clk(clk),
        .wea0(sram_act_wea0),
        .addr0(sram_act_addr0),
        .wdata0(sram_act_wdata0),
        .rdata0(sram_act_rdata0),
        .wea1(sram_act_wea1),
        .addr1(sram_act_addr1),
        .wdata1(sram_act_wdata1),
        .rdata1(sram_act_rdata1)
    );

    softmax softmax_inst(
        .clk(clk),
        .rst(rst_n),
        .data_in_valid(softmax_data_in_valid),
        .data_out_ready(softmax_data_out_ready),
        .in_data(softmax_in_data),
        .in_scale(softmax_in_scale),
        .out_scale(softmax_out_scale),
        .S(sequence_length),
        .data_out_valid(softmax_data_out_valid),
        .data_in_ready(softmax_data_in_ready),
        .out_data(softmax_out_data)
    );

    layernorm layernorm_inst(
        .clk(clk),
        .rst(rst_n),
        .data_in_valid(layernorm_data_in_valid),
        .data_out_ready(layernorm_data_out_ready),
        .in_data(layernorm_in_data),
        .weights(layernorm_weights),
        .bias(layernorm_bias),
        .in_scale(layernorm_in_scale),
        .weight_scale(layernorm_weight_scale),
        .bias_scale(layernorm_bias_scale),
        .out_scale(layernorm_out_scale),
        .data_out_valid(layernorm_data_out_valid),
        .data_in_ready(layernorm_data_in_ready),
        .out_data(layernorm_out_data)
    );

    GELU gelu_inst(
        .clk(clk),
        .rst(rst_n),
        .data_in_valid(gelu_data_in_valid),
        .data_out_ready(gelu_data_out_ready),
        .in_data(gelu_in_data),
        .in_scale(gelu_in_scale),
        .out_scale(gelu_out_scale),
        .data_out_valid(gelu_data_out_valid),
        .data_in_ready(gelu_data_in_ready),
        .out_data(gelu_out_data)
    );


    // ===== Load data ===== //
    initial begin
        // TODO: you should change the filename and sequence length for your own
        weight_sram.load_data("../../pattern/weights/weights.csv");
        act_sram.load_data("../../pattern/patterns/input_0.csv");
        $readmemh("../../pattern/patterns/golden_0.csv", golden);
        sequence_length = 28;
    end


    // ===== System reset ===== //
    initial begin
        clk = 0;
        rst_n = 1;
        compute_start = 0;
        cycle_count = 0;
    end
    
    // ===== Cycle count ===== //
    initial begin
        wait(compute_start == 1);
        start_count = 1;
        wait(compute_finish == 1);
        start_count = 0;
    end

    always @(posedge clk) begin
        if(start_count)
            cycle_count <= cycle_count + 1;
    end 
   
    // ===== Time Exceed Abortion ===== //
    initial begin
        #(`CYCLE*`END_CYCLES);
        $display("\n========================================================");
        $display("You have exceeded the cycle count limit.");
        $display("Simulation abort");
        $display("========================================================");
        $finish;    
    end

    // ===== Clk fliping ===== //
    always #(`CYCLE/2) begin
        clk = ~clk;
    end 

    // ===== Set simulation info ===== //
    initial begin
    `ifdef GATESIM
        $dumpfile("bert_encoder_syn.vcd");
        $dumpvars("+all");
        $sdf_annotate("../../syn/netlist/bert_encoder_syn.sdf", bert_encoder_inst);
	`else
        `ifdef POSTSIM
            $dumpfile("bert_encoder_post.vcd");
            $dumpvars("+all");
            $sdf_annotate("../../apr/netlist/CHIP.sdf", bert_encoder_inst);
        `else
            $dumpfile("bert_encoder.vcd");
            $dumpvars("+all");
        `endif
    `endif
    end
        

    // ===== Simulating  ===== //
    initial begin

        #(`CYCLE*100);
        $display("Reset System");
        @(negedge clk);
        rst_n = 1'b0;
        @(negedge clk);
        @(negedge clk);
        @(negedge clk);
        rst_n = 1'b1;
        $display("Compute start");
        @(negedge clk);
        compute_start = 1'b1;
        @(negedge clk);
        compute_start = 1'b0;

        wait(compute_finish == 1);
        $display("Compute finished, start validating result...");

        validate();

        $display("Simulation finish");
        $finish;
    end

    integer errors, total_errors;
    task validate; begin
        // Input
        total_errors = 0;
        $display("=====================");

        errors = 0;
        for(i=0 ; i<sequence_length*8 ; i=i+1)
            if(golden[i] !== act_sram.RAM[i]) begin
                $display("[ERROR]   [%d] Input Result:%32h Golden:%32h", i, act_sram.RAM[i], golden[i]);
                errors = errors + 1;
            end
            else begin
                // $display("[CORRECT]   [%d] Input Result:%32h Golden:%32h", i, act_sram.RAM[i], golden[i]);
            end
        if(errors == 0)
            $display("Input             [PASS]");
        else
            $display("Input             [FAIL]");
        total_errors = total_errors + errors;

        // FC1
        errors = 0;
        for(i=256 ; i<256+sequence_length*8 ; i=i+1)
            if(golden[i] !== act_sram.RAM[i]) begin
                $display("[ERROR]   [%d] FC1 Result:%32h Golden:%32h", i-256, act_sram.RAM[i], golden[i]);
                errors = errors + 1;
            end
            else begin
                // $display("[CORRECT]   [%d] FC1 Result:%32h Golden:%32h", i-256, act_sram.RAM[i], golden[i]);
            end
        if(errors == 0)
            $display("FC1               [PASS]");
        else
            $display("FC1               [FAIL]");
        total_errors = total_errors + errors;
            
        // Output
        errors = 0;
        for(i=512 ; i<512+sequence_length*8 ; i=i+1)
            if(golden[i] !== act_sram.RAM[i]) begin
                $display("[ERROR]   [%d] Output Result:%32h Golden:%32h", i-512, act_sram.RAM[i], golden[i]);
                errors = errors + 1;
            end
            else begin
                // $display("[CORRECT]   [%d] Output Result:%32h Golden:%32h", i-512, act_sram.RAM[i], golden[i]);
            end
        if(errors == 0)
            $display("Output            [PASS]");
        else
            $display("Output            [FAIL]");
        total_errors = total_errors + errors;
        
        if(total_errors == 0)
            $display(">>> Congratulation! All result are correct");
        else
            $display(">>> There are %d errors QQ", total_errors);
            
    `ifdef GATESIM
        $display("  [Pre-layout gate-level simulation]");
	`else
        `ifdef POSTSIM
            $display("  [Post-layout gate-level simulation]");
        `else
            $display("  [RTL simulation]");
        `endif
    `endif
        $display("Clock Period: %.2f ns,Total cycle count: %d cycles", `CYCLE, cycle_count);
        $display("=====================");
    end
    endtask

endmodule
