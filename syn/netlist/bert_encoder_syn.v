/////////////////////////////////////////////////////////////
// Created by: Synopsys DC Expert(TM) in wire load mode
// Version   : R-2020.09-SP5
// Date      : Wed Jun  5 15:27:36 2024
/////////////////////////////////////////////////////////////


module bert_encoder ( clk, rst_n, compute_start, compute_finish, 
        sequence_length, sram_weight_wea0, sram_weight_addr0, 
        sram_weight_wdata0, sram_weight_rdata0, sram_weight_wea1, 
        sram_weight_addr1, sram_weight_wdata1, sram_weight_rdata1, 
        sram_act_wea0, sram_act_addr0, sram_act_wdata0, sram_act_rdata0, 
        sram_act_wea1, sram_act_addr1, sram_act_wdata1, sram_act_rdata1, 
        softmax_data_in_valid, softmax_data_out_ready, softmax_in_data, 
        softmax_in_scale, softmax_out_scale, softmax_data_out_valid, 
        softmax_data_in_ready, softmax_out_data, layernorm_data_in_valid, 
        layernorm_data_out_ready, layernorm_in_data, layernorm_weights, 
        layernorm_bias, layernorm_in_scale, layernorm_weight_scale, 
        layernorm_bias_scale, layernorm_out_scale, layernorm_data_out_valid, 
        layernorm_data_in_ready, layernorm_out_data, gelu_data_in_valid, 
        gelu_data_out_ready, gelu_in_data, gelu_in_scale, gelu_out_scale, 
        gelu_data_out_valid, gelu_data_in_ready, gelu_out_data );
  input [7:0] sequence_length;
  output [15:0] sram_weight_wea0;
  output [15:0] sram_weight_addr0;
  output [127:0] sram_weight_wdata0;
  input [127:0] sram_weight_rdata0;
  output [15:0] sram_weight_wea1;
  output [15:0] sram_weight_addr1;
  output [127:0] sram_weight_wdata1;
  input [127:0] sram_weight_rdata1;
  output [15:0] sram_act_wea0;
  output [15:0] sram_act_addr0;
  output [127:0] sram_act_wdata0;
  input [127:0] sram_act_rdata0;
  output [15:0] sram_act_wea1;
  output [15:0] sram_act_addr1;
  output [127:0] sram_act_wdata1;
  input [127:0] sram_act_rdata1;
  output [255:0] softmax_in_data;
  output [31:0] softmax_in_scale;
  output [31:0] softmax_out_scale;
  input [255:0] softmax_out_data;
  output [255:0] layernorm_in_data;
  output [255:0] layernorm_weights;
  output [255:0] layernorm_bias;
  output [31:0] layernorm_in_scale;
  output [31:0] layernorm_weight_scale;
  output [31:0] layernorm_bias_scale;
  output [31:0] layernorm_out_scale;
  input [255:0] layernorm_out_data;
  output [255:0] gelu_in_data;
  output [31:0] gelu_in_scale;
  output [31:0] gelu_out_scale;
  input [255:0] gelu_out_data;
  input clk, rst_n, compute_start, softmax_data_out_valid,
         softmax_data_in_ready, layernorm_data_out_valid,
         layernorm_data_in_ready, gelu_data_out_valid, gelu_data_in_ready;
  output compute_finish, softmax_data_in_valid, softmax_data_out_ready,
         layernorm_data_in_valid, layernorm_data_out_ready, gelu_data_in_valid,
         gelu_data_out_ready;


endmodule

