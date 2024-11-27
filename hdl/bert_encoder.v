module bert_encoder (
    input wire clk,
    input wire rst_n,

    input wire compute_start,
    output reg compute_finish,
    input wire [7:0] sequence_length,

    // Weight sram, dual port
    output reg [15:0] sram_weight_wea0,
    output reg [15:0] sram_weight_addr0,
    output reg [127:0] sram_weight_wdata0,
    input wire [127:0] sram_weight_rdata0,
    output reg [15:0] sram_weight_wea1,
    output reg [15:0] sram_weight_addr1,
    output reg [127:0] sram_weight_wdata1,
    input wire [127:0] sram_weight_rdata1,

    // Activation sram, dual port
    output reg [15:0] sram_act_wea0,
    output reg [15:0] sram_act_addr0,
    output reg [127:0] sram_act_wdata0,
    input wire [127:0] sram_act_rdata0,
    output reg [15:0] sram_act_wea1,
    output reg [15:0] sram_act_addr1,
    output reg [127:0] sram_act_wdata1,
    input wire [127:0] sram_act_rdata1,

    // softmax module
    output reg softmax_data_in_valid,
    output reg softmax_data_out_ready,
    output reg [255:0] softmax_in_data,
    output reg [31:0] softmax_in_scale,
    output reg [31:0] softmax_out_scale,
    input wire softmax_data_out_valid,
    input wire softmax_data_in_ready,
    input wire [255:0] softmax_out_data,

    // layernorm module
    output reg layernorm_data_in_valid,
    output reg layernorm_data_out_ready,
    output reg [255:0] layernorm_in_data,
    output reg [255:0] layernorm_weights,
    output reg [255:0] layernorm_bias,
    output reg [31:0] layernorm_in_scale,
    output reg [31:0] layernorm_weight_scale,
    output reg [31:0] layernorm_bias_scale,
    output reg [31:0] layernorm_out_scale,
    input wire layernorm_data_out_valid,
    input wire layernorm_data_in_ready,
    input wire [255:0] layernorm_out_data,

    // GELU module
    output reg gelu_data_in_valid,
    output reg gelu_data_out_ready,
    output reg [255:0] gelu_in_data,
    output reg [31:0] gelu_in_scale,
    output reg [31:0] gelu_out_scale,
    input wire gelu_data_out_valid,
    input wire gelu_data_in_ready,
    input wire [255:0] gelu_out_data
);

// Add your design here
//read data out from sram
//========= input to DFF1 ==========//
reg compute_start_r;
reg compute_finish_r;
reg signed [7:0] sram_act_rdata0_r [15:0]; 

reg signed [7:0] sram_act_rdata1_r [15:0];

reg signed [7:0] sram_weight_rdata0_r [15:0];

reg signed [7:0] sram_weight_rdata1_r [15:0];

reg [7:0] sequence_length_r;




always @(posedge clk) begin
    //input
    compute_start_r <= compute_start;
    sequence_length_r <= sequence_length;

    {sram_act_rdata0_r[15], sram_act_rdata0_r[14], sram_act_rdata0_r[13], sram_act_rdata0_r[12], sram_act_rdata0_r[11], 
     sram_act_rdata0_r[10], sram_act_rdata0_r[9], sram_act_rdata0_r[8], sram_act_rdata0_r[7], sram_act_rdata0_r[6], sram_act_rdata0_r[5], 
     sram_act_rdata0_r[4], sram_act_rdata0_r[3], sram_act_rdata0_r[2], sram_act_rdata0_r[1], sram_act_rdata0_r[0]} <= sram_act_rdata0;

    {sram_act_rdata1_r[15], sram_act_rdata1_r[14], sram_act_rdata1_r[13], sram_act_rdata1_r[12], sram_act_rdata1_r[11],
     sram_act_rdata1_r[10], sram_act_rdata1_r[9], sram_act_rdata1_r[8], sram_act_rdata1_r[7], sram_act_rdata1_r[6], sram_act_rdata1_r[5],
     sram_act_rdata1_r[4], sram_act_rdata1_r[3], sram_act_rdata1_r[2], sram_act_rdata1_r[1], sram_act_rdata1_r[0]} <= sram_act_rdata1;

    {sram_weight_rdata0_r[15], sram_weight_rdata0_r[14], sram_weight_rdata0_r[13], sram_weight_rdata0_r[12], sram_weight_rdata0_r[11],
     sram_weight_rdata0_r[10], sram_weight_rdata0_r[9], sram_weight_rdata0_r[8], sram_weight_rdata0_r[7], sram_weight_rdata0_r[6], sram_weight_rdata0_r[5],
     sram_weight_rdata0_r[4], sram_weight_rdata0_r[3], sram_weight_rdata0_r[2], sram_weight_rdata0_r[1], sram_weight_rdata0_r[0]} <= sram_weight_rdata0;
    
    {sram_weight_rdata1_r[15], sram_weight_rdata1_r[14], sram_weight_rdata1_r[13], sram_weight_rdata1_r[12], sram_weight_rdata1_r[11],
     sram_weight_rdata1_r[10], sram_weight_rdata1_r[9], sram_weight_rdata1_r[8], sram_weight_rdata1_r[7], sram_weight_rdata1_r[6], sram_weight_rdata1_r[5],
     sram_weight_rdata1_r[4], sram_weight_rdata1_r[3], sram_weight_rdata1_r[2], sram_weight_rdata1_r[1], sram_weight_rdata1_r[0]} <= sram_weight_rdata1;
end
//================================//


//============= FSM ==============//
localparam IDLE = 6'b000000;

localparam Scale_read = 6'b000011;  // 0001
localparam Memory_read0 = 6'b000010; // 0010
localparam Memory_read2 = 6'b000100; // 0011
localparam Compute = 6'b000101;  // 00100
localparam Add_Bias = 6'b000110;
localparam Requant = 6'b001000;
localparam Write = 6'b001010;
localparam Normalize = 6'b001100;
localparam Compute1 = 6'b001101; // 01010
localparam finish = 6'b001011;   
localparam softmax_tx = 6'b001110;
localparam softmax_rx0 = 6'b010000;
localparam softmax_complete = 6'b010100;
localparam Compute2 = 6'b011101; // 10000
localparam WAIT = 6'b010110;
localparam Requant1 = 6'b011000;
//Layernorm
localparam Layernorm_read0 = 6'b011010;
localparam Layernorm_read1 = 6'b011100;
localparam Layernorm_requant0 = 6'b011110;//16clk
localparam Layernorm_requant1 = 6'b100000;//32clk
localparam Layernorm_add = 6'b100010;
localparam Layernorm_TX = 6'b100100;
localparam Layernorm_RX = 6'b100110;

//GELU
localparam gelu_tx = 6'b101000;
localparam gelu_rx0 = 6'b101010;
localparam gelu_rx1 = 6'b101100;

localparam Compute3 = 6'b111101 ;


localparam Q = 4'd0; 
localparam K = 4'd1;
localparam V = 4'd2;
localparam Attention_result = 4'd3;
localparam FC1 = 4'd4; 
localparam Layernorm = 4'd5;
localparam FF1 = 4'd6;
localparam FF2 = 4'd7;
localparam FFN = 4'd8;


reg [5:0] current_state , next_state;

reg [6:0] write_cnt;    //write_cnt //0-127

reg [6:0] write_cnt_bbb; //0-127

reg [1:0] compute_cnt;   //compute_cnt 0-3

reg [2:0] write_bias_cnt;   //write_bias_cnt 0-7
reg [4:0] write_bias_cnt_1;   //write_bias_cnt 0-7

reg [9:0] weight_row_cnt;   //weight_row_cnt 0-255

reg [1:0] write_out_cnt;  //write_out_cnt 0-3

reg [10:0] cnt5;

reg [6:0] cnt6;

reg Q1_Q2_cnt;

reg write_out_cnt_bbb;

reg [7:0] input_row_cnt;  //input_row_cnt //0-127
reg [6:0] input_row_cnt_1;

reg signed [31:0] scale;

reg  signed [24:0] requant_input_data;    //requant_input_data


reg signed [7:0] bias;

wire signed [24:0] output_pe;  //output_pe


wire signed [7:0] requant_output;    //requnt 出來的output data 為8bit bu = requant_output


reg [3:0] QKV;
reg [3:0] QKV_next;




//layernorm
reg [3:0] write_cnt_nonlinear , write_cnt_nonlinear_next;//紀錄Compute_Softmax算幾次


reg signed [31:0] scale0_model , scale0_model_next;//for non-linear model
reg signed [31:0] scale1_model , scale1_model_next;//for non-linear model
reg signed [31:0] scale2_model , scale2_model_next;//for non-linear model
reg signed [31:0] scale3_model , scale3_model_next;//for non-linear model
reg [5:0] cnt , cnt_next;

reg signed [8:0] layernorm_buffer0 [31:0];//layernorm module的data
reg signed [7:0] layernorm_buffer1 [31:0];//layernorm module的weight
reg signed [7:0] layernorm_buffer2 [31:0];//layernorm module的bias

reg  signed [7:0] adder_in0 [15:0];//layernorm adder
wire signed [8:0] adder_out [15:0];//layernorm adder



always @(posedge clk) begin
    if(!rst_n) current_state <= IDLE;
    else current_state <= next_state;
end

always @(*) begin
    case(current_state)
        IDLE: begin
            if(compute_start_r) next_state = Scale_read;
            else next_state = IDLE;
        end
        Scale_read: next_state = Memory_read0;
        Memory_read0: next_state = Memory_read2;
        Memory_read2: begin 
            if(QKV == Attention_result) next_state = Compute1;   // Q scale k x
            else if(QKV == Layernorm || QKV == FFN) next_state = Layernorm_requant0;
            else if (QKV == FF2) next_state = Compute3;
            else next_state = Compute;   // Q scale k x 
        end 
        Compute: begin
            if (compute_cnt == 3) next_state = Add_Bias;
            else next_state = Compute;
        end
        Compute3: begin
            if (cnt == 15) next_state = Add_Bias;
            else next_state = Compute3;
        end
        Add_Bias: begin 
            if(write_cnt == 15) next_state = Requant;
            else if (weight_row_cnt == {2'b0 , sequence_length_r} && QKV == 2) next_state = Requant;
            else if (QKV == FF2) next_state = Compute3;
            else next_state = Compute;
        end
        Requant: begin 
            if(QKV == FF1) next_state = gelu_tx;
            else next_state = Write;
        end
        Write: begin  // Q weight addr  =0  K weight addr 12398 
            if (QKV == Attention_result && write_out_cnt !=3) next_state = softmax_rx0;
            else if (QKV == FC1 && input_row_cnt == sequence_length_r) next_state = Layernorm_read0;
            else if (QKV ==  FF2 && input_row_cnt == sequence_length_r) next_state = Layernorm_read0; 
            else next_state = Memory_read0;
        end
        Compute1 : begin  // Q*K
            if (compute_cnt == 1) next_state = Normalize;
            else next_state = Compute1;
        end
        Normalize : begin 
            if(cnt5 == {3'b0 , sequence_length_r}) next_state = Requant1;
            else next_state = Compute1;
        end
        softmax_tx : next_state = softmax_rx0;
        softmax_rx0 : next_state = softmax_complete;
        softmax_complete : next_state = Compute2;
        Compute2 : next_state = WAIT; // *V
        WAIT : begin //V pe = 0
            if(write_cnt == 15) next_state = Requant;
            else next_state = Compute2;
        end
        Requant1: next_state = softmax_tx;
        finish: next_state = IDLE;


        //Layernorm
        Layernorm_read0: next_state = Layernorm_read1;
        Layernorm_read1: next_state = Memory_read0;
        Layernorm_requant0: begin
            if(cnt == 15) next_state = Layernorm_add;
            else next_state = Layernorm_requant0;
        end
        Layernorm_add: next_state = write_cnt_nonlinear[0] ? Layernorm_requant1 : Memory_read0;
        Layernorm_requant1: begin
            if(cnt == 31) next_state = Layernorm_TX;
            else next_state = Layernorm_requant1;
        end
        Layernorm_TX: begin
            if(write_cnt_nonlinear == 8) next_state = layernorm_data_in_ready ? Layernorm_RX : Layernorm_TX;//Layernorm module
            else next_state = layernorm_data_in_ready ? Layernorm_read0 : Layernorm_TX;//暫定分開傳
        end 
        Layernorm_RX: begin
            if(input_row_cnt == sequence_length_r) begin
                if(QKV == Layernorm) next_state = Memory_read0;//要跳到FC1
                else next_state = finish;//要跳到FF2
            end
            else next_state = write_cnt == 5 ? Layernorm_read0 : Layernorm_RX;//RX寫入SRAM
        end 

        //gelu
        gelu_tx: next_state = gelu_rx0;
        gelu_rx0: next_state = gelu_rx1;
        gelu_rx1: next_state = Write;

        default: next_state = IDLE;
    endcase
end

always@(posedge clk)begin
    if(!rst_n) compute_finish <= 1'b0;
    else if(current_state == finish) compute_finish <= 1'b1;
    else compute_finish <= 1'b0;
end

//Write enable 
always@(posedge clk)begin
    if(!rst_n)begin
        sram_act_wea0 <= 16'b0;
        sram_act_wea1 <= 16'b0;
        sram_weight_wea0 <= 16'b0;
        sram_weight_wea1 <= 16'b0;
        sram_weight_wdata1 <= 128'b0;
    end
    else begin
        sram_act_wea1 <= 16'b0;
        sram_weight_wea1 <= 16'b0;
        sram_weight_wdata1 <= 128'b0;
        case(current_state)
            Requant:begin
                case(QKV)
                Q,V,Attention_result,FC1,FF2:begin
                    sram_act_wea0 <= 16'b1111111111111111;
                    sram_weight_wea0 <= 16'b0;
                end
                K:begin
                    sram_act_wea0 <= 16'b0;
                    sram_weight_wea0 <= 16'b1111111111111111;
                end
                default:begin
                    sram_act_wea0 <= 16'b0;
                    sram_weight_wea0 <= 16'b0;
                end
                endcase
            end
            Layernorm_RX: begin
                if(write_cnt == 0 || write_cnt == 5) begin
                    sram_act_wea0 <= 0;
                    sram_act_wea1 <= 0;
                end
                else begin//寫入SRAM
                    sram_act_wea0 <= 16'b1111111111111111;
                    sram_act_wea1 <= 16'b1111111111111111;                    
                end
            end
            gelu_rx1: begin
                sram_act_wea0 <= 16'b1111111111111111;
                sram_weight_wea0 <= 16'b0;
            end
            default: begin
                sram_act_wea0 <= 16'b0;
                sram_weight_wea0 <= 16'b0;
            end
    endcase
    end

end
//================================//
/* 
        counter
*/

always @(posedge clk) begin
    if(!rst_n) cnt <= 0;
    else cnt <= cnt_next;
end

always @(*) begin
    case(QKV)
        Layernorm,FFN: begin//layernorm
            if(current_state == Layernorm_add || current_state == Layernorm_TX) cnt_next = 0;
            else if(current_state == Layernorm_requant0 || current_state == Layernorm_requant1) cnt_next = cnt + 1;
            else cnt_next = cnt;
        end
        FF2:begin
            if(current_state == Compute3) cnt_next = cnt + 1;
            else cnt_next = 0;
        end
        default: begin //暫定
            cnt_next = 0;            
        end
    endcase
end






// =0 Q1 =1 Q2
always@(posedge clk)begin
    if(!rst_n) Q1_Q2_cnt <= 0;
    else if(QKV == Attention_result && input_row_cnt == sequence_length_r) Q1_Q2_cnt <= Q1_Q2_cnt + 1;
    else Q1_Q2_cnt <= Q1_Q2_cnt;
end

always@(posedge clk)begin
    if(!rst_n) cnt5 <= 0;
    else if(QKV == Attention_result && (current_state == Memory_read2 || current_state == Normalize)) begin 
        if(cnt5 == {3'b0 , sequence_length_r}) cnt5 <= 0;
        else cnt5 <= cnt5 + 1;
    end
    else if (QKV == Layernorm) cnt5 <= 0;
    else if ((QKV == FC1 || QKV == FF1 || QKV == FF2) && current_state == Write)begin
        if(input_row_cnt == sequence_length_r) cnt5 <= 0;
        else cnt5 <= cnt5 + 1;
    end
    else cnt5 <= cnt5;
end

always@(posedge clk)begin   //cnt = 16筆 8bit data 處理完，寫到sram 0-15  16只是為了判斷式簡單點 如果要15歸零的話
    if(!rst_n) write_cnt <= 0; //，那就要 if(current_state == Add_Bias && cnt == 15) cnt <= 7'b0; 
    else if (current_state == Write) write_cnt <= 0;
    else if (current_state == Add_Bias) write_cnt <= write_cnt + 1;
    else if (current_state == WAIT)write_cnt <= write_cnt + 1;
    else if (QKV == Layernorm || QKV == FFN) begin
        if(current_state == Layernorm_RX) begin 
            if (input_row_cnt == sequence_length_r) write_cnt <= 0;
            else write_cnt <= write_cnt + 1;//暫定一次傳四筆(4clk)
        end
        else write_cnt <= 0;//暫定歸0
    end
    else write_cnt <= write_cnt;
end

always@(posedge clk)begin   
    if(!rst_n) cnt6 <= 0;
    else if (current_state == WAIT) cnt6 <= cnt6 + 1;
    else if (current_state == Memory_read0) cnt6 <= 0;
    else cnt6 <= cnt6;
end

always@(posedge clk)begin      //compute_cnt = compute state 要算多久 0-3
    if(!rst_n) write_cnt_bbb <= 0;
    else if (write_cnt_bbb == 16 && current_state == Write) write_cnt_bbb <= 0;
    else if (current_state == Add_Bias) begin 
        if(QKV == V) begin 
            if(weight_row_cnt == {2'b0 , sequence_length_r}) write_cnt_bbb <= write_cnt_bbb + 1;
            else write_cnt_bbb <= write_cnt_bbb;
        end
        else write_cnt_bbb <= write_cnt_bbb + 1;
    end
    else write_cnt_bbb <= write_cnt_bbb;
end
always@(posedge clk)begin      //compute_cnt = compute state 要算多久 0-3
    if(!rst_n) compute_cnt <= 0;
    else if(current_state[0]) compute_cnt <= compute_cnt + 1;
    else compute_cnt <= 0;
end
always@(posedge clk)begin      //write_bias_cnt 0-7 : write_bias_cnt = 現在是第幾個地址的bias ex: 1030 1031....
    if(!rst_n) write_bias_cnt <= 0;   // 0-3 寫到q1 4-7寫到q2
    else if (current_state == Requant) begin
        if(QKV == V && write_cnt_bbb == 16) write_bias_cnt <= write_bias_cnt + 1;
        else write_bias_cnt <= write_bias_cnt;
    end
    else if(current_state == Write) begin
        if(QKV ==Q || QKV == K || QKV == FC1 || QKV == FF2) write_bias_cnt <= write_bias_cnt + 1;
        else write_bias_cnt <= write_bias_cnt;
    end
    else write_bias_cnt <= write_bias_cnt;
end
always@(posedge clk)begin      //write_bias_cnt 0-7 : write_bias_cnt = 現在是第幾個地址的bias ex: 1030 1031....
    if(!rst_n) write_bias_cnt_1 <= 0;   // 0-3 寫到q1 4-7寫到q2
    else if(current_state == Write) begin
        if(QKV == FF1) write_bias_cnt_1 <= write_bias_cnt_1 + 1;
        else write_bias_cnt_1 <= write_bias_cnt_1;
    end
    else write_bias_cnt_1 <= write_bias_cnt_1;
end
always@(posedge clk)begin    //weight_row_cnt = 0-127 ，現在weight在第幾row 128代表說我們要進行最後一筆運算
    if(!rst_n) weight_row_cnt <= 0;
    else if (current_state == Compute && compute_cnt == 0) weight_row_cnt <= weight_row_cnt + 1;
    else if (current_state == Compute3 && cnt == 0) weight_row_cnt <= weight_row_cnt + 1;
    else if (current_state == Requant) begin 
        case(QKV)
            V: begin
                if(weight_row_cnt == {2'b0 , sequence_length_r}) weight_row_cnt <= 0;
                else weight_row_cnt <= weight_row_cnt;
            end
            FF1: begin
                if(weight_row_cnt == 512) weight_row_cnt <= 0;
                else weight_row_cnt <= weight_row_cnt;
            end
            default: begin
                if(weight_row_cnt == 128) weight_row_cnt <= 0;
                else weight_row_cnt <= weight_row_cnt;
            end
        endcase
    end
    else if (current_state == Normalize) weight_row_cnt <= weight_row_cnt + 1;
    else if (current_state == softmax_tx) weight_row_cnt <= 0;
    else weight_row_cnt <= weight_row_cnt;
end
always@(posedge clk)begin   //output 現在要寫到哪一個地址  0-3 因為Q1 Q2 K1 K2 然後這次的spec的weight、act sram一個row 就是4個addr，所以write_out_cnt是拿來判斷是寫到哪一個column
    if(!rst_n) write_out_cnt <= 0;
    else if(current_state == Write) write_out_cnt <= write_out_cnt + 1;
    else write_out_cnt <= write_out_cnt;
end
always@(posedge clk)begin   //output 現在要寫到哪一個地址  0-3 因為Q1 Q2 K1 K2 然後這次的spec的weight、act sram一個row 就是4個addr，所以write_out_cnt是拿來判斷是寫到哪一個column
    if(!rst_n) write_out_cnt_bbb <= 0;
    else if(current_state == Write) write_out_cnt_bbb <= write_out_cnt_bbb + 1;
    else write_out_cnt_bbb <= write_out_cnt_bbb;
end
always@(posedge clk)begin    //input_row_cnt input第幾個row 或者拿來判斷現在要寫到哪一個row 
    if(!rst_n) input_row_cnt <= 0;
    else if(current_state == Requant) begin
        case(QKV)
            Q,K,FC1,FF2 : begin
                if(weight_row_cnt == 128) input_row_cnt <= input_row_cnt + 1;
                else input_row_cnt <= input_row_cnt;
            end
            V: begin
                if(weight_row_cnt == {2'b0 , sequence_length_r}) input_row_cnt <= input_row_cnt + 1;
                else input_row_cnt <= input_row_cnt;
            end
            Attention_result: begin
                if(write_out_cnt == 3) input_row_cnt <= input_row_cnt + 1;
                else input_row_cnt <= input_row_cnt;
            end
            FF1: begin
                if(weight_row_cnt == 512) input_row_cnt <= input_row_cnt + 1;
                else input_row_cnt <= input_row_cnt;
            end
            default: input_row_cnt <= 0;
        endcase 
    end
    else if (current_state == Write) begin
        case(QKV)
            V : begin
                if(input_row_cnt == 128) input_row_cnt <= 0;
                else input_row_cnt <= input_row_cnt;
            end
            default : begin
                if(input_row_cnt == sequence_length_r) input_row_cnt <= 0;
                else input_row_cnt <= input_row_cnt;
            end
        endcase
    end
    else if(current_state == Layernorm_RX) begin//
        if(input_row_cnt == sequence_length_r) input_row_cnt <= 0;
        else if(write_cnt == 4) input_row_cnt <= input_row_cnt + 1;
        else input_row_cnt <= input_row_cnt;
    end
    else input_row_cnt <= input_row_cnt;
end
always@(posedge clk)begin    //input_row_cnt input第幾個row 或者拿來判斷現在要寫到哪一個row 
    if(!rst_n) input_row_cnt_1 <= 8'b0;
    else if(current_state == Requant && QKV == 3 && write_out_cnt == 3) input_row_cnt_1 <= input_row_cnt_1 + 8'b1;
    else if (QKV == FC1 && input_row_cnt == sequence_length_r) input_row_cnt_1 <= 0;
    else if(current_state == Layernorm_RX) begin//
        if(input_row_cnt_1 == sequence_length_r[6:0]) input_row_cnt_1 <= 0;
        else if(write_cnt == 3) input_row_cnt_1 <= input_row_cnt_1 + 1;
        else input_row_cnt_1 <= input_row_cnt_1;
    end
    else input_row_cnt_1 <= input_row_cnt_1;
end

//////////////////////////////////////////////////////

 //                        GELU
//////////////////////////////////////////////////////
always@(posedge clk)begin
    if(!rst_n) gelu_in_data <= 128'b0;
    else begin
        case(write_cnt)
        7'd1: gelu_in_data[7:0] <= requant_output;
        7'd2: gelu_in_data[15:8] <= requant_output;
        7'd3: gelu_in_data[23:16] <= requant_output;
        7'd4: gelu_in_data[31:24] <= requant_output;
        7'd5: gelu_in_data[39:32] <= requant_output;
        7'd6: gelu_in_data[47:40] <= requant_output;
        7'd7: gelu_in_data[55:48] <= requant_output;
        7'd8: gelu_in_data[63:56] <= requant_output;
        7'd9: gelu_in_data[71:64] <= requant_output;
        7'd10: gelu_in_data[79:72] <= requant_output;
        7'd11: gelu_in_data[87:80] <= requant_output;
        7'd12: gelu_in_data[95:88] <= requant_output;
        7'd13: gelu_in_data[103:96] <= requant_output;
        7'd14: gelu_in_data[111:104] <= requant_output;
        7'd15: gelu_in_data[119:112] <= requant_output;
        7'd16: gelu_in_data[127:120] <= requant_output;
        default: gelu_in_data <= 128'b0;
        endcase
    end
end

always@(*)begin
    if(current_state == gelu_tx) gelu_data_in_valid = 1'b1;
    else gelu_data_in_valid = 1'b0;
end

always@(*)begin
    case(current_state)
    gelu_tx , gelu_rx0,gelu_rx1: gelu_data_out_ready = 1'b1;
    default: gelu_data_out_ready = 1'b0;
    endcase
end

always@(*)begin 
    gelu_in_scale = scale0_model;
    gelu_out_scale = scale1_model;
end





///////////////////////////////////////////////
//Non-linear model--Softmax
always @(posedge clk) begin
    if(!rst_n) write_cnt_nonlinear <= 0;
    else write_cnt_nonlinear <= write_cnt_nonlinear_next;
end

always @(*) begin
    case(current_state)
        Layernorm_add: write_cnt_nonlinear_next = write_cnt_nonlinear + 1;//8組8個16bit element才能進入Layernorm
        Layernorm_RX: write_cnt_nonlinear_next = 0;//暫定
        default: write_cnt_nonlinear_next = write_cnt_nonlinear;
    endcase 
end


//////////////////////////////////////////////////////////////////////



always@(*)begin
    if(current_state == softmax_tx) softmax_data_in_valid = 1'b1;
    else softmax_data_in_valid = 1'b0;
end

always@(*)begin
    case(current_state)
    softmax_tx , softmax_rx0,softmax_complete: softmax_data_out_ready = 1'b1;
    default: softmax_data_out_ready = 1'b0;
    endcase
end

always@(*)begin 
    softmax_in_scale = {sram_weight_rdata0_r[3],sram_weight_rdata0_r[2],sram_weight_rdata0_r[1],sram_weight_rdata0_r[0]};
    softmax_out_scale = {sram_weight_rdata0_r[7],sram_weight_rdata0_r[6],sram_weight_rdata0_r[5],sram_weight_rdata0_r[4]};
end


always@(posedge clk)begin
    if(!rst_n) begin 
        sram_weight_wdata0 <= 128'b0;
        sram_act_wdata0 <= 128'b0;
        sram_act_wdata1 <= 0;
    end
    else if(QKV == Layernorm || QKV == FFN) begin//layernorm
        sram_weight_wdata0 <= 128'b0;
        if(current_state == Layernorm_RX) begin
            sram_act_wdata0[127:0] <= layernorm_out_data[127:0];
            sram_act_wdata1[127:0] <= layernorm_out_data[255:128];
        end
        else begin
            sram_act_wdata0[127:0] <= 0;
            sram_act_wdata1[127:0] <= 0;            
        end
    end
    else if(QKV == FF1)begin
        sram_act_wdata0 <= gelu_out_data[127:0];
        sram_weight_wdata0 <= 0;
        sram_act_wdata1 <= 0;
    end 
    else begin
        sram_act_wdata1 <= 0;
        case(write_cnt) 
            7'd1: begin 
                sram_weight_wdata0[7:0] <= requant_output;
                sram_act_wdata0[7:0] <= requant_output;
            end
            7'd2: begin 
                sram_weight_wdata0[15:8] <= requant_output;
                sram_act_wdata0[15:8] <= requant_output;
            end
            7'd3: begin 
                sram_weight_wdata0[23:16] <= requant_output;
                sram_act_wdata0[23:16] <= requant_output;
            end
            7'd4: begin 
                sram_weight_wdata0[31:24] <= requant_output;
                sram_act_wdata0[31:24] <= requant_output;
            end
            7'd5: begin 
                sram_weight_wdata0[39:32] <= requant_output;
                sram_act_wdata0[39:32] <= requant_output;
            end
            7'd6: begin 
                sram_weight_wdata0[47:40] <= requant_output;
                sram_act_wdata0[47:40] <= requant_output;
            end
            7'd7: begin 
                sram_weight_wdata0[55:48] <= requant_output;
                sram_act_wdata0[55:48] <= requant_output;
            end
            7'd8: begin 
                sram_weight_wdata0[63:56] <= requant_output;
                sram_act_wdata0[63:56] <= requant_output;
            end
            7'd9: begin 
                sram_weight_wdata0[71:64] <= requant_output;
                sram_act_wdata0[71:64] <= requant_output;
            end
            7'd10: begin 
                sram_weight_wdata0[79:72] <= requant_output;
                sram_act_wdata0[79:72] <= requant_output;
            end
            7'd11: begin 
                sram_weight_wdata0[87:80] <= requant_output;
                sram_act_wdata0[87:80] <= requant_output;
            end
            7'd12: begin 
                sram_weight_wdata0[95:88] <= requant_output;
                sram_act_wdata0[95:88] <= requant_output;
            end
            7'd13: begin 
                sram_weight_wdata0[103:96] <= requant_output;
                sram_act_wdata0[103:96] <= requant_output;
            end
            7'd14: begin 
                sram_weight_wdata0[111:104] <= requant_output;
                sram_act_wdata0[111:104] <= requant_output;
            end
            7'd15: begin 
                sram_weight_wdata0[119:112] <= requant_output;
                sram_act_wdata0[119:112] <= requant_output;
            end
            7'd16: begin 
                sram_weight_wdata0[127:120] <= requant_output;
                sram_act_wdata0[127:120] <= requant_output;
            end 
            default: begin 
                sram_weight_wdata0 <= 128'b0;
                sram_act_wdata0 <= 128'b0;
            end
        endcase
    end
end


always@(posedge clk)begin
    if(!rst_n) begin 
        softmax_in_data <= 256'b0;
    end
    else begin
        case(weight_row_cnt)
            10'd1: softmax_in_data[7:0] <= requant_output;
            10'd2: softmax_in_data[15:8] <= requant_output;
            10'd3: softmax_in_data[23:16] <= requant_output;
            10'd4: softmax_in_data[31:24] <= requant_output;
            10'd5: softmax_in_data[39:32] <= requant_output;
            10'd6: softmax_in_data[47:40] <= requant_output;
            10'd7: softmax_in_data[55:48] <= requant_output;
            10'd8: softmax_in_data[63:56] <= requant_output;
            10'd9: softmax_in_data[71:64] <= requant_output;
            10'd10: softmax_in_data[79:72] <= requant_output;
            10'd11: softmax_in_data[87:80] <= requant_output;
            10'd12: softmax_in_data[95:88] <= requant_output;
            10'd13: softmax_in_data[103:96] <= requant_output;
            10'd14: softmax_in_data[111:104] <= requant_output;
            10'd15: softmax_in_data[119:112] <= requant_output;
            10'd16: softmax_in_data[127:120] <= requant_output;
            10'd17: softmax_in_data[135:128] <= requant_output;
            10'd18: softmax_in_data[143:136] <= requant_output;
            10'd19: softmax_in_data[151:144] <= requant_output;
            10'd20: softmax_in_data[159:152] <= requant_output;
            10'd21: softmax_in_data[167:160] <= requant_output;
            10'd22: softmax_in_data[175:168] <= requant_output;
            10'd23: softmax_in_data[183:176] <= requant_output;
            10'd24: softmax_in_data[191:184] <= requant_output;
            10'd25: softmax_in_data[199:192] <= requant_output;
            10'd26: softmax_in_data[207:200] <= requant_output;
            10'd27: softmax_in_data[215:208] <= requant_output;
            10'd28: softmax_in_data[223:216] <= requant_output;
            10'd29: softmax_in_data[231:224] <= requant_output;
            10'd30: softmax_in_data[239:232] <= requant_output;
            10'd31: softmax_in_data[247:240] <= requant_output;
            10'd32: softmax_in_data[255:248] <= requant_output;
            default: softmax_in_data <= 256'b0;
        endcase
    end
end
always@(*)begin
    case(write_cnt_bbb)
        7'd0: bias = sram_weight_rdata0_r[0];
        7'd1: bias = sram_weight_rdata0_r[1];
        7'd2: bias = sram_weight_rdata0_r[2];
        7'd3: bias = sram_weight_rdata0_r[3];
        7'd4: bias = sram_weight_rdata0_r[4];
        7'd5: bias = sram_weight_rdata0_r[5];
        7'd6: bias = sram_weight_rdata0_r[6];
        7'd7: bias = sram_weight_rdata0_r[7];
        7'd8: bias = sram_weight_rdata0_r[8];
        7'd9: bias = sram_weight_rdata0_r[9];
        7'd10: bias = sram_weight_rdata0_r[10];
        7'd11: bias = sram_weight_rdata0_r[11];
        7'd12: bias = sram_weight_rdata0_r[12];
        7'd13: bias = sram_weight_rdata0_r[13];
        7'd14: bias = sram_weight_rdata0_r[14];
        7'd15: bias = sram_weight_rdata0_r[15];
        default: bias = 0;
    endcase
end
always@(posedge clk)begin  //requant_input_data = requant input data
    if(!rst_n) requant_input_data <= 32'b0;
    else if(current_state == Add_Bias) requant_input_data <= bias + output_pe;
    else if (current_state == Normalize) requant_input_data <= (output_pe>>>3);
    else if (current_state == WAIT) requant_input_data <= output_pe;

    else if(QKV == Layernorm || QKV == FFN)begin
        if(current_state == Memory_read2) requant_input_data  <= {{17{sram_act_rdata0_r[0][7]}} , sram_act_rdata0_r[0]};
        else if(current_state == Layernorm_requant0) begin
                case(cnt)
                    0: requant_input_data  <= {{17{sram_act_rdata0_r[1][7]}} , sram_act_rdata0_r[1]};
                    1: requant_input_data  <= {{17{sram_act_rdata0_r[2][7]}} , sram_act_rdata0_r[2]};
                    2: requant_input_data  <= {{17{sram_act_rdata0_r[3][7]}} , sram_act_rdata0_r[3]};
                    3: requant_input_data  <= {{17{sram_act_rdata0_r[4][7]}} , sram_act_rdata0_r[4]};
                    4: requant_input_data  <= {{17{sram_act_rdata0_r[5][7]}} , sram_act_rdata0_r[5]};
                    5: requant_input_data  <= {{17{sram_act_rdata0_r[6][7]}} , sram_act_rdata0_r[6]};
                    6: requant_input_data  <= {{17{sram_act_rdata0_r[7][7]}} , sram_act_rdata0_r[7]};
                    7: requant_input_data  <= {{17{sram_act_rdata0_r[8][7]}} , sram_act_rdata0_r[8]};
                    8: requant_input_data  <= {{17{sram_act_rdata0_r[9][7]}} , sram_act_rdata0_r[9]};
                    9: requant_input_data  <= {{17{sram_act_rdata0_r[10][7]}} , sram_act_rdata0_r[10]};
                    10: requant_input_data <= {{17{sram_act_rdata0_r[11][7]}} , sram_act_rdata0_r[11]};
                    11: requant_input_data <= {{17{sram_act_rdata0_r[12][7]}} , sram_act_rdata0_r[12]};
                    12: requant_input_data <= {{17{sram_act_rdata0_r[13][7]}} , sram_act_rdata0_r[13]};
                    13: requant_input_data <= {{17{sram_act_rdata0_r[14][7]}} , sram_act_rdata0_r[14]};
                    14: requant_input_data <= {{17{sram_act_rdata0_r[15][7]}} , sram_act_rdata0_r[15]};

                    default: requant_input_data <= requant_input_data;
                endcase
        end
        else if(current_state == Layernorm_add) begin
                requant_input_data <= {{16{layernorm_buffer0[0][8]}} , layernorm_buffer0[0]};
        end
        else if(current_state == Layernorm_requant1) begin
                case(cnt)
                    0:  requant_input_data <= {{16{layernorm_buffer0[1][8]}} , layernorm_buffer0[1]};
                    1:  requant_input_data <= {{16{layernorm_buffer0[2][8]}} , layernorm_buffer0[2]};
                    2:  requant_input_data <= {{16{layernorm_buffer0[3][8]}} , layernorm_buffer0[3]};
                    3:  requant_input_data <= {{16{layernorm_buffer0[4][8]}} , layernorm_buffer0[4]};
                    4:  requant_input_data <= {{16{layernorm_buffer0[5][8]}} , layernorm_buffer0[5]};
                    5:  requant_input_data <= {{16{layernorm_buffer0[6][8]}} , layernorm_buffer0[6]};
                    6:  requant_input_data <= {{16{layernorm_buffer0[7][8]}} , layernorm_buffer0[7]};
                    7:  requant_input_data <= {{16{layernorm_buffer0[8][8]}} , layernorm_buffer0[8]};
                    8:  requant_input_data <= {{16{layernorm_buffer0[9][8]}} , layernorm_buffer0[9]};
                    9:  requant_input_data <= {{16{layernorm_buffer0[10][8]}} , layernorm_buffer0[10]};
                    10: requant_input_data <= {{16{layernorm_buffer0[11][8]}} , layernorm_buffer0[11]};
                    11: requant_input_data <= {{16{layernorm_buffer0[12][8]}} , layernorm_buffer0[12]};
                    12: requant_input_data <= {{16{layernorm_buffer0[13][8]}} , layernorm_buffer0[13]};
                    13: requant_input_data <= {{16{layernorm_buffer0[14][8]}} , layernorm_buffer0[14]};
                    14: requant_input_data <= {{16{layernorm_buffer0[15][8]}} , layernorm_buffer0[15]};
                    15: requant_input_data <= {{16{layernorm_buffer0[16][8]}} , layernorm_buffer0[16]};
                    16: requant_input_data <= {{16{layernorm_buffer0[17][8]}} , layernorm_buffer0[17]};
                    17: requant_input_data <= {{16{layernorm_buffer0[18][8]}} , layernorm_buffer0[18]};
                    18: requant_input_data <= {{16{layernorm_buffer0[19][8]}} , layernorm_buffer0[19]};
                    19: requant_input_data <= {{16{layernorm_buffer0[20][8]}} , layernorm_buffer0[20]};
                    20: requant_input_data <= {{16{layernorm_buffer0[21][8]}} , layernorm_buffer0[21]};
                    21: requant_input_data <= {{16{layernorm_buffer0[22][8]}} , layernorm_buffer0[22]};
                    22: requant_input_data <= {{16{layernorm_buffer0[23][8]}} , layernorm_buffer0[23]};
                    23: requant_input_data <= {{16{layernorm_buffer0[24][8]}} , layernorm_buffer0[24]};
                    24: requant_input_data <= {{16{layernorm_buffer0[25][8]}} , layernorm_buffer0[25]};
                    25: requant_input_data <= {{16{layernorm_buffer0[26][8]}} , layernorm_buffer0[26]};
                    26: requant_input_data <= {{16{layernorm_buffer0[27][8]}} , layernorm_buffer0[27]};
                    27: requant_input_data <= {{16{layernorm_buffer0[28][8]}} , layernorm_buffer0[28]};
                    28: requant_input_data <= {{16{layernorm_buffer0[29][8]}} , layernorm_buffer0[29]};
                    29: requant_input_data <= {{16{layernorm_buffer0[30][8]}} , layernorm_buffer0[30]};
                    30: requant_input_data <= {{16{layernorm_buffer0[31][8]}} , layernorm_buffer0[31]};
                    default: requant_input_data <= requant_input_data;
                endcase
        end
        else begin
                requant_input_data <= requant_input_data;                
        end
    end
    else requant_input_data <= requant_input_data;
end



always@(posedge clk)begin
    if(!rst_n)begin
        sram_act_addr0 <= 16'b0;
        sram_act_addr1 <= 16'b0;
    end
    else if(current_state == Scale_read)begin
        sram_act_addr0 <= 16'b0;
        sram_act_addr1 <= 16'd1;
    end
    else if (current_state == Memory_read0 || current_state == Memory_read2)begin
        if(QKV == Layernorm || QKV == FFN) begin//layernorm維持
                sram_act_addr0 <= sram_act_addr0;
                sram_act_addr1 <= sram_act_addr1;  
            end
        else begin//QKV , softmax都一樣 
                sram_act_addr0 <= sram_act_addr0 + 2;
                sram_act_addr1 <= sram_act_addr1 + 2;                 
        end
    end
    else if (current_state == gelu_rx1)begin
        sram_act_addr0 <= 16'd1792 + cnt5;
        sram_act_addr1 <=0;
    end
    else if (current_state == softmax_tx)begin
        if(Q1_Q2_cnt == 1)begin
            if(sequence_length_r<17)begin
                sram_act_addr0 <= 16'd1088;
                sram_act_addr1 <= 16'd1089;
            end
            else begin
                sram_act_addr0 <= 16'd1152;
                sram_act_addr1 <= 16'd1153;
            end
        end
        else begin
            sram_act_addr0 <= 16'd1024;
            sram_act_addr1 <= 16'd1025;
        end
    end
    else if (current_state == softmax_rx0)begin
        sram_act_addr0 <= sram_act_addr0;
        sram_act_addr1 <= sram_act_addr1;
    end
    else if (current_state == Write)begin
        if(input_row_cnt == sequence_length_r)begin
            if(QKV == 3)begin
                if(Q1_Q2_cnt == 1)begin
                    sram_act_addr0 <= 16'd1280;
                    sram_act_addr1 <= 16'd1281;
                end
                else begin
                    sram_act_addr0 <= 16'd768 + (sequence_length_r <<2);
                    sram_act_addr1 <= 16'd769 + (sequence_length_r <<2);
                end
                
            end
            else if(QKV == 2) begin
                sram_act_addr0 <= 16'd0 + (weight_row_cnt<<3);
                sram_act_addr1 <= 16'd1 + (weight_row_cnt<<3);
            end
            else if (QKV == 4)begin
                sram_act_addr0 <= 0;
                sram_act_addr1 <= 256;
            end
            else if (QKV == 6)begin
                sram_act_addr0 <= 1792;
                sram_act_addr1 <= 1793;
            end
            else if (QKV == 7)begin
                sram_act_addr0 <= 1536;
                sram_act_addr1 <= 2816;
            end
            else begin
                sram_act_addr0 <= 16'd0;
                sram_act_addr1 <= 16'd1;
            end
        end
        else if (input_row_cnt == 128)begin
            sram_act_addr0 <= 16'd768;
            sram_act_addr1 <= 16'd769;
        end
        else if(QKV == 2) begin
            sram_act_addr0 <= 16'd0 + (weight_row_cnt <<3);
            sram_act_addr1 <= 16'd1 + (weight_row_cnt <<3);
        end
        else if (QKV== 3)begin
            if(write_out_cnt == 3)begin
                sram_act_addr0 <= 16'd768 + (input_row_cnt_1 <<2);
                sram_act_addr1 <= 16'd769 + (input_row_cnt_1 <<2);
            end
            else begin
                if(Q1_Q2_cnt == 0) begin
                    if(sequence_length_r<17)begin
                        sram_act_addr0 <= 16'd1024 + cnt6;
                        sram_act_addr1 <= 16'd1025 + cnt6;
                    end
                    else begin
                        sram_act_addr0 <= 16'd1024 + (cnt6 <<1);
                        sram_act_addr1 <= 16'd1025 + (cnt6 <<1);
                    end
                end
                else begin
                    if(sequence_length_r<17)begin
                        sram_act_addr0 <= 16'd1088 + cnt6;
                        sram_act_addr1 <= 16'd1089 + cnt6;
                    end
                    else begin
                        sram_act_addr0 <= 16'd1152 + (cnt6 <<1);
                        sram_act_addr1 <= 16'd1153 + (cnt6 <<1);
                    end
                end
            end 
        end
        else if (QKV == 4)begin
            sram_act_addr0 <= 16'd1280 + (input_row_cnt <<3);
            sram_act_addr1 <= 16'd1281 + (input_row_cnt <<3);
        end
        else if (QKV == 6)begin
            sram_act_addr0 <= 16'd1536 + (input_row_cnt <<3);
            sram_act_addr1 <= 16'd1537 + (input_row_cnt <<3);
        end
        else if (QKV == 7)begin
            sram_act_addr0 <= 16'd1792 + (input_row_cnt <<5);
            sram_act_addr1 <= 16'd1793 + (input_row_cnt <<5);
        end
        else begin
            sram_act_addr0 <= 16'd0 + (input_row_cnt <<3);
            sram_act_addr1 <= 16'd1 + (input_row_cnt <<3);
        end
    end
    else if (current_state == Requant)begin
        if(QKV == 0)begin
            if(write_bias_cnt<4)begin //0-3 Q1的地址
                sram_act_addr0 <= 16'd768 + write_out_cnt + (input_row_cnt <<2);
                sram_act_addr1 <= 16'd0;
            end
            else begin   //4-7 Q2 的地址      //write_out_cnt 0-3 
                sram_act_addr0 <= 16'd768 + write_out_cnt + (sequence_length_r <<2) + (input_row_cnt <<2);
                sram_act_addr1 <= 16'd0;
            end
        end
        else if(QKV == 2)begin
                if(sequence_length_r < 8'd17 ) sram_act_addr0 <= 16'd1024 + input_row_cnt;
                else sram_act_addr0 <= 16'd1024 + write_out_cnt_bbb + (input_row_cnt <<1);
                sram_act_addr1 <= 16'd0;
        end
        else if (QKV == 3)begin
            if(Q1_Q2_cnt == 0)begin 
                sram_act_addr0 <= 16'd1280 + write_out_cnt + (input_row_cnt <<3);
                sram_act_addr1 <= 16'd0;
            end
            else begin
                sram_act_addr0 <= 16'd1284 + write_out_cnt + (input_row_cnt <<3);
                sram_act_addr1 <= 16'd0;
            end
        end
        else if(QKV == 4)begin
            sram_act_addr0 <= 16'd256 + cnt5;
            sram_act_addr1 <=0;
        end
        else if(QKV == 7)begin
            sram_act_addr0 <= 16'd2816 + cnt5;
            sram_act_addr1 <=0;
        end
        else begin
            sram_act_addr0 <= sram_act_addr0;
            sram_act_addr1 <= sram_act_addr1;
        end
    end
    else if (current_state == Compute)begin
        if(compute_cnt == 2)begin
            if (QKV == 2) begin
                sram_act_addr0 <= 16'd0 + (weight_row_cnt << 3);
                sram_act_addr1 <= 16'd1 + (weight_row_cnt << 3);
            end
            else if (QKV == 4)begin
                sram_act_addr0 <= 16'd1280 + (input_row_cnt <<3);
                sram_act_addr1 <= 16'd1281 + (input_row_cnt <<3);
            end
            else if (QKV == 6)begin
                sram_act_addr0 <= 16'd1536 + (input_row_cnt <<3);
                sram_act_addr1 <= 16'd1537 + (input_row_cnt <<3);
            end 
            else begin
                sram_act_addr0 <= 16'd0 + (input_row_cnt <<3);
                sram_act_addr1 <= 16'd1 + (input_row_cnt <<3);
            end
        end
        else begin
            sram_act_addr0 <= sram_act_addr0 + 16'd2;
            sram_act_addr1 <= sram_act_addr1 + 16'd2;
        end
    end
    else if (current_state == Compute1)begin
        if(compute_cnt == 0)begin
            sram_act_addr0 <= 768 + (input_row_cnt_1 <<2);
            sram_act_addr1 <= 769 + (input_row_cnt_1 <<2);
        end
        else begin
            sram_act_addr0 <= sram_act_addr0 + 16'd2;
            sram_act_addr1 <= sram_act_addr1 + 16'd2;
        end
    end
    else if (current_state == Compute2) begin
        sram_act_addr0 <= sram_act_addr0;
        sram_act_addr1 <= sram_act_addr1;
    end
    else if (current_state == Compute3)begin
        if(cnt == 14)begin
                sram_act_addr0 <= 1792 + (input_row_cnt <<5);
                sram_act_addr1 <= 1793 + (input_row_cnt <<5);
        end
        else begin
            sram_act_addr0 <= sram_act_addr0 + 16'd2;
            sram_act_addr1 <= sram_act_addr1 + 16'd2;
        end
    end
    else if (current_state == Layernorm_add || current_state == Layernorm_requant1 || current_state == Layernorm_TX || current_state == Layernorm_read0 || current_state == Layernorm_read1 ) begin
        sram_act_addr0 <= sram_act_addr0;
        sram_act_addr1 <= sram_act_addr1;
    end
    else if(current_state == Layernorm_RX) begin
        if(input_row_cnt == sequence_length_r) begin//暫定FF1
            sram_act_addr0 <= 1536;
            sram_act_addr1 <= 1537;            
        end
        else begin
            if(write_cnt == 5) begin
                if(QKV == Layernorm)begin
                    sram_act_addr0 <= (input_row_cnt << 3);//回到layernorm_read0
                    sram_act_addr1 <= (input_row_cnt << 3) + 256; 
                end
                else begin
                    sram_act_addr0 <= 1536 + (input_row_cnt << 3);//回到layernorm_read0
                    sram_act_addr1 <= (input_row_cnt << 3) + 2816; 
                end
                                  
            end
            else if(layernorm_data_out_valid) begin//此時寫進SRAM
                if(write_cnt == 1) begin
                    if(QKV == Layernorm)begin
                        sram_act_addr0 <= 1536 + (input_row_cnt << 3);
                        sram_act_addr1 <= 1537 + (input_row_cnt << 3); 
                    end
                    else begin
                        sram_act_addr0 <= 512 + (input_row_cnt << 3);
                        sram_act_addr1 <= 513 + (input_row_cnt << 3); 
                    end                   
                end
                else begin
                    sram_act_addr0 <= sram_act_addr0 + 2;
                    sram_act_addr1 <= sram_act_addr1 + 2;                    
                end
            end
            else begin//RX最一開始還沒傳layernorm_out_data
                sram_act_addr0 <= sram_act_addr0;
                sram_act_addr1 <= sram_act_addr1;              
            end
        end
    end
    else if (current_state == Layernorm_requant0 || current_state == Layernorm_requant1)begin
        if(cnt == 15)begin
            sram_act_addr0 <= sram_act_addr0 + 1;
            sram_act_addr1 <= sram_act_addr1 + 1;
        end
        else begin
            sram_act_addr0 <= sram_act_addr0;
            sram_act_addr1 <= sram_act_addr1;
        end
    end
    else if ((current_state == 20 || current_state == 22) && sequence_length_r < 17)begin
        sram_act_addr0 <= sram_act_addr0 + 16'd1;
        sram_act_addr1 <= 0;
    end
    else begin
        sram_act_addr0 <= sram_act_addr0 + 16'd2;
        sram_act_addr1 <= sram_act_addr1 + 16'd2; 
    end  
end

always@(posedge clk)begin
    if(!rst_n)begin
        sram_weight_addr0 <= 16'b0;
        sram_weight_addr1 <= 16'b0;
    end
    else if(current_state == IDLE)begin
        sram_weight_addr0 <= 16'b0;
        sram_weight_addr1 <= 16'd0;
    end
    else if(current_state == Scale_read)begin
        sram_weight_addr0 <= 16'd6;
        sram_weight_addr1 <= 16'd7;
    end
    else if (current_state == Memory_read0)begin
        if(QKV == 5 || QKV == FFN) begin//layernorm
                sram_weight_addr0 <= sram_weight_addr0;
                sram_weight_addr1 <= sram_weight_addr1;                
            end
        else begin
                sram_weight_addr0 <= sram_weight_addr0 + 2;
                sram_weight_addr1 <= sram_weight_addr1 + 2;                
        end
    end
    else if (current_state == Memory_read2)begin
        if(QKV == 3) begin
            sram_weight_addr0 <= sram_weight_addr0;
            sram_weight_addr1 <= sram_weight_addr1;
        end
        else if (QKV == 5 || QKV == FFN)begin
            sram_weight_addr0 <= sram_weight_addr0;
            sram_weight_addr1 <= sram_weight_addr1;
        end
        else begin
            sram_weight_addr0 <= sram_weight_addr0 + 2;
            sram_weight_addr1 <= sram_weight_addr1 + 2;
        end
    end
    else if (current_state == gelu_rx1)begin
        sram_weight_addr0 <= 16'd3;
        sram_weight_addr1 <= 16'd4;
    end
    else if (current_state == Write)begin
        if (QKV == 0)begin
            if(input_row_cnt == sequence_length_r) begin
                sram_weight_addr0 <= 16'd1038;
                sram_weight_addr1 <= 16'd1039;
            end
            else begin
                sram_weight_addr0 <= 16'd6 + (weight_row_cnt <<3);
                sram_weight_addr1 <= 16'd7 + (weight_row_cnt <<3);
            end
        end
        else if (QKV == 1)begin
            if(input_row_cnt == sequence_length_r)begin
                sram_weight_addr0 <= 16'd2070 ;
                sram_weight_addr1 <= 16'd2071 ;
            end
            else begin
                sram_weight_addr0 <= 16'd1038 + (weight_row_cnt <<3);
                sram_weight_addr1 <= 16'd1039 + (weight_row_cnt <<3);
            end
        end
        else if (QKV == 2)begin
            if(input_row_cnt == 128)begin
                sram_weight_addr0 <= 16'd12398;
                sram_weight_addr1 <= 16'd12399;
            end
            else begin
                sram_weight_addr0 <= 16'd2070 + (input_row_cnt <<3);
                sram_weight_addr1 <= 16'd2071 + (input_row_cnt <<3);
            end
        end
        else if (QKV == 3)begin
            if(input_row_cnt == sequence_length_r)begin
                if(Q1_Q2_cnt == 0)begin
                    sram_weight_addr0 <= 16'd12398 + (sequence_length_r <<2);
                    sram_weight_addr1 <= 16'd12399 + (sequence_length_r <<2);
                end
                else begin
                    sram_weight_addr0 <= 16'd3102;
                    sram_weight_addr1 <= 16'd3103;
                end
            end
            else if (Q1_Q2_cnt == 1)begin
                sram_weight_addr0 <= 16'd12398 + (sequence_length_r <<2);
                sram_weight_addr1 <= 16'd12399 + (sequence_length_r <<2);
            end
            else begin
                sram_weight_addr0 <= 16'd12398;
                sram_weight_addr1 <= 16'd12399;
            end  
        end
        else if (QKV == 4)begin
            if(input_row_cnt == sequence_length_r) begin
                sram_weight_addr0 <= 16'd4134; //layernorm1 weight
                sram_weight_addr1 <= 16'd4142; //layernorm1 bias
            end
            else begin
                sram_weight_addr0 <= 16'd3102 + (weight_row_cnt <<3);
                sram_weight_addr1 <= 16'd3103 + (weight_row_cnt <<3);
            end
        end
        else if (QKV == 6)begin
            if(input_row_cnt == sequence_length_r) begin
                sram_weight_addr0 <= 16'd8278; 
                sram_weight_addr1 <= 16'd8279; 
            end
            else begin
                sram_weight_addr0 <= 16'd4150 + (weight_row_cnt <<3);
                sram_weight_addr1 <= 16'd4151 + (weight_row_cnt <<3);
            end
        end
        else if (QKV == 7)begin
            if(input_row_cnt == sequence_length_r) begin
                sram_weight_addr0 <= 16'd12382; 
                sram_weight_addr1 <= 16'd12390; 
            end
            else begin
                sram_weight_addr0 <= 16'd8278 + (weight_row_cnt <<5);
                sram_weight_addr1 <= 16'd8279 + (weight_row_cnt <<5);
            end
        end
    end
    else if (current_state == Requant)begin
            if(QKV == 3)begin
                if(write_out_cnt == 3) sram_weight_addr1 <= 0;
                else sram_weight_addr1 <= 1;
                sram_weight_addr0 <= 1;
            end
            else if (QKV == 4)begin
                sram_weight_addr0 <= 1;
                sram_weight_addr1 <= 2;
            end
            else if (QKV == 7)begin
                sram_weight_addr0 <= 3;
                sram_weight_addr1 <= 4;
            end
            else if(write_bias_cnt<4)begin                      
                sram_weight_addr0 <= 16'd12398 + write_out_cnt + (input_row_cnt <<2);
                sram_weight_addr1 <= 16'd0;
            end
            else if (weight_row_cnt == {2'b0  , sequence_length_r} && input_row_cnt == 127)begin
                sram_weight_addr0 <= 16'd0;
                sram_weight_addr1 <= 16'd0;
            end
            else begin
                sram_weight_addr0 <= 16'd12398 + write_out_cnt + (sequence_length_r <<2) + (input_row_cnt <<2);
                sram_weight_addr1 <= 16'd0;
            end
    end
    else if (current_state == Normalize)begin
        sram_weight_addr0 <= sram_weight_addr0;
        sram_weight_addr1 <= sram_weight_addr1;
    end
    else if (current_state == Requant1)begin
        sram_weight_addr0 <= 16'd0;
        sram_weight_addr1 <= 16'd1;
    end
    else if (current_state == Compute1)begin
        if(cnt5 == {3'b0  , sequence_length_r}) sram_weight_addr0 <= 1;
        else sram_weight_addr0 <= sram_weight_addr0 + 2;
        sram_weight_addr1 <= sram_weight_addr1 + 2;
    end
    else if (current_state == Compute)begin
        if(compute_cnt ==1)begin
            sram_weight_addr1 <= 16'd0;
            if(QKV == 0)begin
                sram_weight_addr0 <= 16'd1030 + write_bias_cnt;
            end
            else if (QKV == 1)begin
                sram_weight_addr0 <= 16'd2062 + write_bias_cnt;
            end
            else if (QKV == 2)begin
                sram_weight_addr0 <= 16'd3094 + write_bias_cnt;
            end
            else if (QKV == 4)begin
                sram_weight_addr0 <= 16'd4126 + write_bias_cnt;
            end
            else if (QKV == 6)begin
                sram_weight_addr0 <= 16'd8246 + write_bias_cnt_1;
            end
        end
        else if (compute_cnt == 2)begin
            if(QKV == 0)begin
                sram_weight_addr0 <= 16'd6 + 8 * weight_row_cnt ;
                sram_weight_addr1 <= 16'd7 + 8 * weight_row_cnt ;
            end
            else if (QKV == 1)begin
                sram_weight_addr0 <= 16'd1038 + 8 * weight_row_cnt ;
                sram_weight_addr1 <= 16'd1039 + 8 * weight_row_cnt ;
            end
            else if (QKV == 2)begin
                sram_weight_addr0 <= 16'd2070 +  (input_row_cnt <<3) ;
                sram_weight_addr1 <= 16'd2071 +  (input_row_cnt <<3) ;
            end
            else if (QKV == 4)begin
                sram_weight_addr0 <= 16'd3102 +  (weight_row_cnt<<3) ;
                sram_weight_addr1 <= 16'd3103 +  (weight_row_cnt<<3) ;
            end
            else if (QKV == 6)begin
                sram_weight_addr0 <= 16'd4150 +  (weight_row_cnt<<3) ;
                sram_weight_addr1 <= 16'd4151 +  (weight_row_cnt<<3) ;
            end
        end
        else begin
            sram_weight_addr0 <= sram_weight_addr0 + 16'd2;
            sram_weight_addr1 <= sram_weight_addr1 + 16'd2;
        end
    end
    else if (current_state == Compute3)begin
        if(cnt == 13)begin
            sram_weight_addr0 <= 16'd12374 + write_bias_cnt;
            sram_weight_addr1 <= 16'd0;
        end
        else if (cnt == 14)begin
            sram_weight_addr0 <= 16'd8278 +  (weight_row_cnt<<5) ;
            sram_weight_addr1 <= 16'd8279 +  (weight_row_cnt<<5) ;
        end
        else begin
            sram_weight_addr0 <= sram_weight_addr0 + 16'd2;
            sram_weight_addr1 <= sram_weight_addr1 + 16'd2;
        end
    end
    else if (current_state == Layernorm_read0)begin
        sram_weight_addr0 <= sram_weight_addr0 + 1;
        sram_weight_addr1 <= sram_weight_addr1 + 1;
    end
    else if (current_state == Layernorm_read1)begin
        if(QKV == Layernorm)begin
            sram_weight_addr0 <= 3;
            sram_weight_addr1 <= 2;
        end
        else begin
            sram_weight_addr0 <= 5;
            sram_weight_addr1 <= 4;
        end
    end
    else if(current_state == Layernorm_RX)begin
        if(input_row_cnt_1 == sequence_length_r[6:0]) begin
            if (QKV == Layernorm)begin 
            sram_weight_addr0 <= 3;
            sram_weight_addr1 <= 4;
            end
            else begin
                sram_weight_addr0 <= 4;
                sram_weight_addr1 <= 5;
            end
        end
        else if (input_row_cnt == sequence_length_r)begin
            sram_weight_addr0 <= 4150;
            sram_weight_addr1 <= 4151;
        end
        else if (write_cnt == 4)begin
            if(QKV == Layernorm)begin
                sram_weight_addr0 <= 3;
                sram_weight_addr1 <= 2;
            end
            else begin
                sram_weight_addr0 <= 5;
                sram_weight_addr1 <= 4;
            end
        end
        else begin
            if(QKV == Layernorm)begin
                sram_weight_addr0 <= 4134 + write_cnt_nonlinear;
                sram_weight_addr1 <= 4142 + write_cnt_nonlinear;
            end
            else begin
                sram_weight_addr0 <= 12382 + write_cnt_nonlinear;
                sram_weight_addr1 <= 12390 + write_cnt_nonlinear;
            end
            
        end       
    end
    else if (current_state == Layernorm_TX)begin
        if(QKV == Layernorm)begin
                sram_weight_addr0 <= 4134 + write_cnt_nonlinear;
                sram_weight_addr1 <= 4142 + write_cnt_nonlinear;
            end
        else begin
                sram_weight_addr0 <= 12382 + write_cnt_nonlinear;
                sram_weight_addr1 <= 12390 + write_cnt_nonlinear;
        end
    end
    else if (current_state == Layernorm_requant0 || current_state == Layernorm_requant1 || current_state == Layernorm_add) begin
        sram_weight_addr0 <= sram_weight_addr0;
        sram_weight_addr1 <= sram_weight_addr1;
    end
    else begin
        sram_weight_addr0 <= sram_weight_addr0 + 16'd2;
        sram_weight_addr1 <= sram_weight_addr1 + 16'd2;
    end

end

always@(posedge clk)begin            //讀scale data 的地方
    if(!rst_n) scale <= 32'b0;
    else if(current_state == Memory_read2) begin
        if (QKV == Q) scale <= {sram_weight_rdata1_r[3] , sram_weight_rdata1_r[2] , sram_weight_rdata1_r[1] , sram_weight_rdata1_r[0]};
        else if (QKV == K) scale <= {sram_weight_rdata1_r[7] , sram_weight_rdata1_r[6] , sram_weight_rdata1_r[5] , sram_weight_rdata1_r[4]};
        else if (QKV == V) scale <= {sram_weight_rdata1_r[11] , sram_weight_rdata1_r[10] , sram_weight_rdata1_r[9] , sram_weight_rdata1_r[8]};
        else if (QKV == Attention_result) scale <= {sram_weight_rdata1_r[15] , sram_weight_rdata1_r[14] , sram_weight_rdata1_r[13] , sram_weight_rdata1_r[12]};
        else if (QKV == FC1) scale <= {sram_weight_rdata0_r[15] , sram_weight_rdata0_r[14] , sram_weight_rdata0_r[13] , sram_weight_rdata0_r[12]};
        else if (QKV == FF1) scale <= {sram_weight_rdata0_r[11] , sram_weight_rdata0_r[10] , sram_weight_rdata0_r[9] , sram_weight_rdata0_r[8]};
        else if (QKV == FF2) scale <= {sram_weight_rdata1_r[7] , sram_weight_rdata1_r[6] , sram_weight_rdata1_r[5] , sram_weight_rdata1_r[4]};
        else scale <= scale;
    end
    else if (current_state == softmax_complete) scale <= {sram_weight_rdata1_r[11] , sram_weight_rdata1_r[10] , sram_weight_rdata1_r[9] , sram_weight_rdata1_r[8]}; 
    else if (current_state == Layernorm_read1) begin 
        if(QKV == Layernorm) scale <= {sram_weight_rdata1_r[3] , sram_weight_rdata1_r[2] , sram_weight_rdata1_r[1] , sram_weight_rdata1_r[0]};  //residual scale
        else scale <= {sram_weight_rdata1_r[11] , sram_weight_rdata1_r[10] , sram_weight_rdata1_r[9] , sram_weight_rdata1_r[8]};   
    end
    else if (current_state == Layernorm_add)begin
        if (write_cnt_nonlinear[0] == 0) begin 
            if(QKV == Layernorm) scale <= {sram_weight_rdata1_r[3] , sram_weight_rdata1_r[2] , sram_weight_rdata1_r[1] , sram_weight_rdata1_r[0]}; //residual scale
            else scale <= {sram_weight_rdata1_r[11] , sram_weight_rdata1_r[10] , sram_weight_rdata1_r[9] , sram_weight_rdata1_r[8]}; 
        end
        else begin
            if(QKV == Layernorm) scale <= {sram_weight_rdata1_r[7] , sram_weight_rdata1_r[6] , sram_weight_rdata1_r[5] , sram_weight_rdata1_r[4]}; //add scale
            else scale <= {sram_weight_rdata1_r[15] , sram_weight_rdata1_r[14] , sram_weight_rdata1_r[13] , sram_weight_rdata1_r[12]}; 
        end
    end
    else scale <= scale;
end



always @(posedge clk) begin
    if(!rst_n) begin
        scale0_model <= 0;
        scale1_model <= 0;
        scale2_model <= 0;
        scale3_model <= 0;
    end
    else begin
        scale0_model <= scale0_model_next;
        scale1_model <= scale1_model_next;
        scale2_model <= scale2_model_next;
        scale3_model <= scale3_model_next;      
    end
end

always @(*) begin
    case(QKV)
        Layernorm: begin//layernorm
            if(current_state == Layernorm_add) begin//哪個state都可以，因為我固定地址了
                scale0_model_next = {sram_weight_rdata1_r[11] , sram_weight_rdata1_r[10] , sram_weight_rdata1_r[9] , sram_weight_rdata1_r[8]};
                scale1_model_next = {sram_weight_rdata1_r[15] , sram_weight_rdata1_r[14] , sram_weight_rdata1_r[13] , sram_weight_rdata1_r[12]};
                scale2_model_next = {sram_weight_rdata0_r[3] , sram_weight_rdata0_r[2] , sram_weight_rdata0_r[1] , sram_weight_rdata0_r[0]};
                scale3_model_next = {sram_weight_rdata0_r[7] , sram_weight_rdata0_r[6] , sram_weight_rdata0_r[5] , sram_weight_rdata0_r[4]};                 
            end
            else begin
                scale0_model_next = scale0_model;
                scale1_model_next = scale1_model;
                scale2_model_next = scale2_model;
                scale3_model_next = scale3_model;                
            end
        end
        FF1: begin
            if(current_state == Memory_read2)begin
                scale0_model_next = {sram_weight_rdata0_r[15] , sram_weight_rdata0_r[14] , sram_weight_rdata0_r[13] , sram_weight_rdata0_r[12]};
                scale1_model_next = {sram_weight_rdata1_r[3] , sram_weight_rdata1_r[2] , sram_weight_rdata1_r[1] , sram_weight_rdata1_r[0]};
                scale2_model_next = scale2_model;
                scale3_model_next = scale3_model;
            end
            else begin
                scale0_model_next = scale0_model;
                scale1_model_next = scale1_model;
                scale2_model_next = scale2_model;
                scale3_model_next = scale3_model;                
            end            
        end
        FFN: begin//layernorm
            if(current_state == Layernorm_add) begin//哪個state都可以，因為我固定地址了
                scale0_model_next = {sram_weight_rdata0_r[3] , sram_weight_rdata0_r[2] , sram_weight_rdata0_r[1] , sram_weight_rdata0_r[0]};
                scale1_model_next = {sram_weight_rdata0_r[7] , sram_weight_rdata0_r[6] , sram_weight_rdata0_r[5] , sram_weight_rdata0_r[4]};
                scale2_model_next = {sram_weight_rdata0_r[11] , sram_weight_rdata0_r[10] , sram_weight_rdata0_r[9] , sram_weight_rdata0_r[8]};
                scale3_model_next = {sram_weight_rdata0_r[15] , sram_weight_rdata0_r[14] , sram_weight_rdata0_r[13] , sram_weight_rdata0_r[12]};                 
            end
            else begin
                scale0_model_next = scale0_model;
                scale1_model_next = scale1_model;
                scale2_model_next = scale2_model;
                scale3_model_next = scale3_model;                
            end
        end
        default: begin
            scale0_model_next = scale0_model;
            scale1_model_next = scale1_model;  
            scale2_model_next = scale2_model;
            scale3_model_next = scale3_model;           
        end
    endcase
end

always@(*)begin
    case(QKV)
        Q: begin
            if(input_row_cnt == sequence_length_r) QKV_next = K;
            else QKV_next = Q;
        end
        K: begin
            if(input_row_cnt == sequence_length_r) QKV_next = V;
            else QKV_next = K;
        end
        V: begin
            if(input_row_cnt == 128) QKV_next = Attention_result;
            else QKV_next = V;
        end
        Attention_result: begin
            if(input_row_cnt == sequence_length_r && Q1_Q2_cnt == 1) QKV_next = FC1;
            else QKV_next = Attention_result;
        end
        FC1: begin
            if(input_row_cnt == sequence_length_r) QKV_next = Layernorm;
            else QKV_next = FC1;
        end
        Layernorm: begin
            if(input_row_cnt == sequence_length_r) QKV_next = FF1;
            else QKV_next = Layernorm;
        end
        FF1: begin
            if(input_row_cnt == sequence_length_r && current_state == Write) QKV_next = FF2;
            else QKV_next = FF1;
        end
        FF2: begin
            if(input_row_cnt == sequence_length_r) QKV_next = FFN;
            else QKV_next = FF2;
        end
        FFN: begin
            QKV_next = FFN;
        end
        default QKV_next = Q;
    endcase
end
always@(posedge clk)begin
    if(!rst_n) QKV <= Q;
    else QKV <= QKV_next;
end

wire signed [7:0] weight_mux [31:0];
reg signed [7:0] softmax_buffer_out [31:0];
integer  i;
always@(posedge clk)begin
    if(!rst_n)begin
        for(i=0 ;i<32 ; i=i+1)softmax_buffer_out[i] <= 8'b0;
    end
    else if (softmax_data_out_valid)begin
        softmax_buffer_out[0] <= softmax_out_data[7:0];
        softmax_buffer_out[1] <= softmax_out_data[15:8];
        softmax_buffer_out[2] <= softmax_out_data[23:16];
        softmax_buffer_out[3] <= softmax_out_data[31:24];
        softmax_buffer_out[4] <= softmax_out_data[39:32];
        softmax_buffer_out[5] <= softmax_out_data[47:40];
        softmax_buffer_out[6] <= softmax_out_data[55:48];
        softmax_buffer_out[7] <= softmax_out_data[63:56];
        softmax_buffer_out[8] <= softmax_out_data[71:64];
        softmax_buffer_out[9] <= softmax_out_data[79:72];
        softmax_buffer_out[10] <= softmax_out_data[87:80];
        softmax_buffer_out[11] <= softmax_out_data[95:88];
        softmax_buffer_out[12] <= softmax_out_data[103:96];
        softmax_buffer_out[13] <= softmax_out_data[111:104];
        softmax_buffer_out[14] <= softmax_out_data[119:112];
        softmax_buffer_out[15] <= softmax_out_data[127:120];
        softmax_buffer_out[16] <= softmax_out_data[135:128];
        softmax_buffer_out[17] <= softmax_out_data[143:136];
        softmax_buffer_out[18] <= softmax_out_data[151:144];
        softmax_buffer_out[19] <= softmax_out_data[159:152];
        softmax_buffer_out[20] <= softmax_out_data[167:160];
        softmax_buffer_out[21] <= softmax_out_data[175:168];
        softmax_buffer_out[22] <= softmax_out_data[183:176];
        softmax_buffer_out[23] <= softmax_out_data[191:184];
        softmax_buffer_out[24] <= softmax_out_data[199:192];
        softmax_buffer_out[25] <= softmax_out_data[207:200];
        softmax_buffer_out[26] <= softmax_out_data[215:208];
        softmax_buffer_out[27] <= softmax_out_data[223:216];
        softmax_buffer_out[28] <= softmax_out_data[231:224];
        softmax_buffer_out[29] <= softmax_out_data[239:232];
        softmax_buffer_out[30] <= softmax_out_data[247:240];
        softmax_buffer_out[31] <= softmax_out_data[255:248];
    end
    else begin
        for(i=0 ;i<32 ; i=i+1)softmax_buffer_out[i] <= softmax_buffer_out[i];
    end


end
assign weight_mux[0] = (current_state == Compute2) ? softmax_buffer_out[0] : sram_weight_rdata0_r[0];
assign weight_mux[1] = (current_state == Compute2) ? softmax_buffer_out[1] : sram_weight_rdata0_r[1];
assign weight_mux[2] = (current_state == Compute2) ? softmax_buffer_out[2] : sram_weight_rdata0_r[2];
assign weight_mux[3] = (current_state == Compute2) ? softmax_buffer_out[3] : sram_weight_rdata0_r[3];
assign weight_mux[4] = (current_state == Compute2) ? softmax_buffer_out[4] : sram_weight_rdata0_r[4];
assign weight_mux[5] = (current_state == Compute2) ? softmax_buffer_out[5] : sram_weight_rdata0_r[5];
assign weight_mux[6] = (current_state == Compute2) ? softmax_buffer_out[6] : sram_weight_rdata0_r[6];
assign weight_mux[7] = (current_state == Compute2) ? softmax_buffer_out[7] : sram_weight_rdata0_r[7];
assign weight_mux[8] = (current_state == Compute2) ? softmax_buffer_out[8] : sram_weight_rdata0_r[8];
assign weight_mux[9] = (current_state == Compute2) ? softmax_buffer_out[9] : sram_weight_rdata0_r[9];
assign weight_mux[10] = (current_state == Compute2) ? softmax_buffer_out[10] : sram_weight_rdata0_r[10];
assign weight_mux[11] = (current_state == Compute2) ? softmax_buffer_out[11] : sram_weight_rdata0_r[11];
assign weight_mux[12] = (current_state == Compute2) ? softmax_buffer_out[12] : sram_weight_rdata0_r[12];
assign weight_mux[13] = (current_state == Compute2) ? softmax_buffer_out[13] : sram_weight_rdata0_r[13];
assign weight_mux[14] = (current_state == Compute2) ? softmax_buffer_out[14] : sram_weight_rdata0_r[14];
assign weight_mux[15] = (current_state == Compute2) ? softmax_buffer_out[15] : sram_weight_rdata0_r[15];
assign weight_mux[16] = (current_state == Compute2) ? softmax_buffer_out[16] : sram_weight_rdata1_r[0];
assign weight_mux[17] = (current_state == Compute2) ? softmax_buffer_out[17] : sram_weight_rdata1_r[1];
assign weight_mux[18] = (current_state == Compute2) ? softmax_buffer_out[18] : sram_weight_rdata1_r[2];
assign weight_mux[19] = (current_state == Compute2) ? softmax_buffer_out[19] : sram_weight_rdata1_r[3];
assign weight_mux[20] = (current_state == Compute2) ? softmax_buffer_out[20] : sram_weight_rdata1_r[4];
assign weight_mux[21] = (current_state == Compute2) ? softmax_buffer_out[21] : sram_weight_rdata1_r[5];
assign weight_mux[22] = (current_state == Compute2) ? softmax_buffer_out[22] : sram_weight_rdata1_r[6];
assign weight_mux[23] = (current_state == Compute2) ? softmax_buffer_out[23] : sram_weight_rdata1_r[7];
assign weight_mux[24] = (current_state == Compute2) ? softmax_buffer_out[24] : sram_weight_rdata1_r[8];
assign weight_mux[25] = (current_state == Compute2) ? softmax_buffer_out[25] : sram_weight_rdata1_r[9];
assign weight_mux[26] = (current_state == Compute2) ? softmax_buffer_out[26] : sram_weight_rdata1_r[10];
assign weight_mux[27] = (current_state == Compute2) ? softmax_buffer_out[27] : sram_weight_rdata1_r[11];
assign weight_mux[28] = (current_state == Compute2) ? softmax_buffer_out[28] : sram_weight_rdata1_r[12];
assign weight_mux[29] = (current_state == Compute2) ? softmax_buffer_out[29] : sram_weight_rdata1_r[13];
assign weight_mux[30] = (current_state == Compute2) ? softmax_buffer_out[30] : sram_weight_rdata1_r[14];
assign weight_mux[31] = (current_state == Compute2) ? softmax_buffer_out[31] : sram_weight_rdata1_r[15];


pe pe0(
    .clk(clk),
    .rst_n(rst_n),
    .en(current_state[0]),
    .in0(sram_act_rdata0_r[0]),
    .in1(sram_act_rdata0_r[1]),
    .in2(sram_act_rdata0_r[2]),
    .in3(sram_act_rdata0_r[3]),
    .in4(sram_act_rdata0_r[4]),
    .in5(sram_act_rdata0_r[5]),
    .in6(sram_act_rdata0_r[6]),
    .in7(sram_act_rdata0_r[7]),
    .in8(sram_act_rdata0_r[8]),
    .in9(sram_act_rdata0_r[9]),
    .in10(sram_act_rdata0_r[10]),
    .in11(sram_act_rdata0_r[11]),
    .in12(sram_act_rdata0_r[12]),
    .in13(sram_act_rdata0_r[13]),
    .in14(sram_act_rdata0_r[14]),
    .in15(sram_act_rdata0_r[15]),
    .in16(sram_act_rdata1_r[0]),
    .in17(sram_act_rdata1_r[1]),
    .in18(sram_act_rdata1_r[2]),
    .in19(sram_act_rdata1_r[3]),
    .in20(sram_act_rdata1_r[4]),
    .in21(sram_act_rdata1_r[5]),
    .in22(sram_act_rdata1_r[6]),
    .in23(sram_act_rdata1_r[7]),
    .in24(sram_act_rdata1_r[8]),
    .in25(sram_act_rdata1_r[9]),
    .in26(sram_act_rdata1_r[10]),
    .in27(sram_act_rdata1_r[11]),
    .in28(sram_act_rdata1_r[12]),
    .in29(sram_act_rdata1_r[13]),
    .in30(sram_act_rdata1_r[14]),
    .in31(sram_act_rdata1_r[15]),
    .weight0(weight_mux[0]),
    .weight1(weight_mux[1]),
    .weight2(weight_mux[2]),
    .weight3(weight_mux[3]),
    .weight4(weight_mux[4]),
    .weight5(weight_mux[5]),
    .weight6(weight_mux[6]),
    .weight7(weight_mux[7]),
    .weight8(weight_mux[8]),
    .weight9(weight_mux[9]),
    .weight10(weight_mux[10]),
    .weight11(weight_mux[11]),
    .weight12(weight_mux[12]),
    .weight13(weight_mux[13]),
    .weight14(weight_mux[14]),
    .weight15(weight_mux[15]),
    .weight16(weight_mux[16]),
    .weight17(weight_mux[17]),
    .weight18(weight_mux[18]),
    .weight19(weight_mux[19]),
    .weight20(weight_mux[20]),
    .weight21(weight_mux[21]),
    .weight22(weight_mux[22]),
    .weight23(weight_mux[23]),
    .weight24(weight_mux[24]),
    .weight25(weight_mux[25]),
    .weight26(weight_mux[26]),
    .weight27(weight_mux[27]),
    .weight28(weight_mux[28]),
    .weight29(weight_mux[29]),
    .weight30(weight_mux[30]),
    .weight31(weight_mux[31]),
    .out(output_pe)
);

requantization re0(
    .in_data(requant_input_data),
    .scale(scale[24:0]),
    .out_data(requant_output)
);





//================ Layernorm module ================//
//Layernorm bufer存data要給Layernorm module
always @(posedge clk) begin
    if(!rst_n) begin
        for(i = 0; i < 32; i = i + 1) begin
            layernorm_buffer0[i] <= 0;
            layernorm_buffer1[i] <= 0;
            layernorm_buffer2[i] <= 0;
        end
    end
    else begin
        if(current_state == Layernorm_requant0) begin//attention_residual scale
            if(write_cnt_nonlinear[1:0] == 2'b00 || write_cnt_nonlinear[1:0] == 2'b10) begin
                case(cnt)
                    0:  layernorm_buffer0[0]  <= {requant_output[7] , requant_output};
                    1:  layernorm_buffer0[1]  <= {requant_output[7] , requant_output};
                    2:  layernorm_buffer0[2]  <= {requant_output[7] , requant_output};
                    3:  layernorm_buffer0[3]  <= {requant_output[7] , requant_output};
                    4:  layernorm_buffer0[4]  <= {requant_output[7] , requant_output};
                    5:  layernorm_buffer0[5]  <= {requant_output[7] , requant_output};
                    6:  layernorm_buffer0[6]  <= {requant_output[7] , requant_output};
                    7:  layernorm_buffer0[7]  <= {requant_output[7] , requant_output};
                    8:  layernorm_buffer0[8]  <= {requant_output[7] , requant_output};
                    9:  layernorm_buffer0[9]  <= {requant_output[7] , requant_output};
                    10: layernorm_buffer0[10] <= {requant_output[7] , requant_output};
                    11: layernorm_buffer0[11] <= {requant_output[7] , requant_output};
                    12: layernorm_buffer0[12] <= {requant_output[7] , requant_output};
                    13: layernorm_buffer0[13] <= {requant_output[7] , requant_output};
                    14: layernorm_buffer0[14] <= {requant_output[7] , requant_output};
                    15: layernorm_buffer0[15] <= {requant_output[7] , requant_output};
                    default: begin
                        for(i = 0; i < 32; i = i + 1) begin
                            layernorm_buffer0[i] <= layernorm_buffer0[i];
                        end
                    end
                endcase
            end
            else if(write_cnt_nonlinear[1:0] == 2'b01 || write_cnt_nonlinear[1:0] == 2'b11) begin
                case(cnt)
                    0:  layernorm_buffer0[16] <= {requant_output[7] , requant_output};
                    1:  layernorm_buffer0[17] <= {requant_output[7] , requant_output};
                    2:  layernorm_buffer0[18] <= {requant_output[7] , requant_output};
                    3:  layernorm_buffer0[19] <= {requant_output[7] , requant_output};
                    4:  layernorm_buffer0[20] <= {requant_output[7] , requant_output};
                    5:  layernorm_buffer0[21] <= {requant_output[7] , requant_output};
                    6:  layernorm_buffer0[22] <= {requant_output[7] , requant_output};
                    7:  layernorm_buffer0[23] <= {requant_output[7] , requant_output};
                    8:  layernorm_buffer0[24] <= {requant_output[7] , requant_output};
                    9:  layernorm_buffer0[25] <= {requant_output[7] , requant_output};
                    10: layernorm_buffer0[26] <= {requant_output[7] , requant_output};
                    11: layernorm_buffer0[27] <= {requant_output[7] , requant_output};
                    12: layernorm_buffer0[28] <= {requant_output[7] , requant_output};
                    13: layernorm_buffer0[29] <= {requant_output[7] , requant_output};
                    14: layernorm_buffer0[30] <= {requant_output[7] , requant_output};
                    15: layernorm_buffer0[31] <= {requant_output[7] , requant_output};
                    default: begin
                        for(i = 0; i < 31; i = i + 1) begin
                            layernorm_buffer0[i] <= layernorm_buffer0[i];
                        end
                    end
                endcase
            end
            else begin
                for(i = 0; i < 32; i = i + 1) begin//維持
                    layernorm_buffer0[i] <= layernorm_buffer0[i];
                end
            end
        end  
        else if(current_state == Layernorm_add) begin//Layernorm_add狀態
            if(write_cnt_nonlinear[1:0] == 2'b00 || write_cnt_nonlinear[1:0] == 2'b10) begin //0,2,4,6
                for(i = 0; i < 16; i = i + 1) begin
                    layernorm_buffer0[i] <= adder_out[i];
                end
            end
            else begin //if(write_cnt_nonlinear[1:0] == 2'b01 || write_cnt_nonlinear[1:0] == 2'b11) begin
                for(i = 0; i < 16; i = i + 1) begin
                    layernorm_buffer0[i+16] <= adder_out[i];
                end
            end           
        end
        else if(current_state == Layernorm_requant1) begin//attention_add scale
            case(cnt)
                0:  layernorm_buffer0[0] <= {requant_output[7] , requant_output};
                1:  layernorm_buffer0[1] <= {requant_output[7] , requant_output};
                2:  layernorm_buffer0[2] <= {requant_output[7] , requant_output};
                3:  layernorm_buffer0[3] <= {requant_output[7] , requant_output};
                4:  layernorm_buffer0[4] <= {requant_output[7] , requant_output};
                5:  layernorm_buffer0[5] <= {requant_output[7] , requant_output};
                6:  layernorm_buffer0[6] <= {requant_output[7] , requant_output};
                7:  layernorm_buffer0[7] <= {requant_output[7] , requant_output};
                8:  layernorm_buffer0[8] <= {requant_output[7] , requant_output};
                9:  layernorm_buffer0[9] <= {requant_output[7] , requant_output};
                10: layernorm_buffer0[10] <= {requant_output[7] , requant_output};
                11: layernorm_buffer0[11] <= {requant_output[7] , requant_output};
                12: layernorm_buffer0[12] <= {requant_output[7] , requant_output};
                13: layernorm_buffer0[13] <= {requant_output[7] , requant_output};
                14: layernorm_buffer0[14] <= {requant_output[7] , requant_output};
                15: layernorm_buffer0[15] <= {requant_output[7] , requant_output};
                16: layernorm_buffer0[16] <= {requant_output[7] , requant_output};
                17: layernorm_buffer0[17] <= {requant_output[7] , requant_output};
                18: layernorm_buffer0[18] <= {requant_output[7] , requant_output};
                19: layernorm_buffer0[19] <= {requant_output[7] , requant_output};
                20: layernorm_buffer0[20] <= {requant_output[7] , requant_output};
                21: layernorm_buffer0[21] <= {requant_output[7] , requant_output};
                22: layernorm_buffer0[22] <= {requant_output[7] , requant_output};
                23: layernorm_buffer0[23] <= {requant_output[7] , requant_output};
                24: layernorm_buffer0[24] <= {requant_output[7] , requant_output};
                25: layernorm_buffer0[25] <= {requant_output[7] , requant_output};
                26: layernorm_buffer0[26] <= {requant_output[7] , requant_output};
                27: layernorm_buffer0[27] <= {requant_output[7] , requant_output};
                28: layernorm_buffer0[28] <= {requant_output[7] , requant_output};
                29: layernorm_buffer0[29] <= {requant_output[7] , requant_output};
                30: layernorm_buffer0[30] <= {requant_output[7] , requant_output};
                31: layernorm_buffer0[31] <= {requant_output[7] , requant_output};
            endcase
        end   
        else if(current_state == Memory_read0 && write_cnt_nonlinear[0] == 0) begin
            for(i = 0; i < 16; i = i + 1) begin
                layernorm_buffer1[i] <= sram_weight_rdata0_r[i];//放weight
                layernorm_buffer2[i] <= sram_weight_rdata1_r[i];//放bias
            end
        end 
        else if(current_state == Memory_read2 && write_cnt_nonlinear[0] == 0) begin
            for(i = 0; i < 16; i = i + 1) begin
                layernorm_buffer1[i+16] <= sram_weight_rdata0_r[i];//放weight
                layernorm_buffer2[i+16] <= sram_weight_rdata1_r[i];//放bias
            end            
        end
        else begin//暫定維持
            for(i = 0; i < 16; i = i + 1) begin
                layernorm_buffer0[i] <= layernorm_buffer0[i];
                layernorm_buffer1[i] <= layernorm_buffer1[i];
                layernorm_buffer2[i] <= layernorm_buffer2[i];
            end            
        end
    end
end



//adder input判斷
always @(*) begin
    case(write_cnt_nonlinear[1:0])
        2'b00 , 2'b10: begin
            for(i = 0; i < 16; i = i + 1) begin
                adder_in0[i] = layernorm_buffer0[i][7:0];
            end
        end
        2'b01: begin
            for(i = 0; i < 16; i = i + 1) begin
                adder_in0[i] = layernorm_buffer0[i+16][7:0];
            end
        end
        default: begin//2'b11
            for(i = 0; i < 16; i = i + 1) begin
                adder_in0[i] = layernorm_buffer0[i+16];
            end            
        end
    endcase
end

Adder_8bit A0(.in0(adder_in0[0]),.in1(sram_act_rdata1_r[0]),.out(adder_out[0]));
Adder_8bit A1(.in0(adder_in0[1]),.in1(sram_act_rdata1_r[1]),.out(adder_out[1]));
Adder_8bit A2(.in0(adder_in0[2]),.in1(sram_act_rdata1_r[2]),.out(adder_out[2]));
Adder_8bit A3(.in0(adder_in0[3]),.in1(sram_act_rdata1_r[3]),.out(adder_out[3]));
Adder_8bit A4(.in0(adder_in0[4]),.in1(sram_act_rdata1_r[4]),.out(adder_out[4]));
Adder_8bit A5(.in0(adder_in0[5]),.in1(sram_act_rdata1_r[5]),.out(adder_out[5]));
Adder_8bit A6(.in0(adder_in0[6]),.in1(sram_act_rdata1_r[6]),.out(adder_out[6]));
Adder_8bit A7(.in0(adder_in0[7]),.in1(sram_act_rdata1_r[7]),.out(adder_out[7]));
Adder_8bit A8(.in0(adder_in0[8]),.in1(sram_act_rdata1_r[8]),.out(adder_out[8]));
Adder_8bit A9(.in0(adder_in0[9]),.in1(sram_act_rdata1_r[9]),.out(adder_out[9]));
Adder_8bit A10(.in0(adder_in0[10]),.in1(sram_act_rdata1_r[10]),.out(adder_out[10]));
Adder_8bit A11(.in0(adder_in0[11]),.in1(sram_act_rdata1_r[11]),.out(adder_out[11]));
Adder_8bit A12(.in0(adder_in0[12]),.in1(sram_act_rdata1_r[12]),.out(adder_out[12]));
Adder_8bit A13(.in0(adder_in0[13]),.in1(sram_act_rdata1_r[13]),.out(adder_out[13]));
Adder_8bit A14(.in0(adder_in0[14]),.in1(sram_act_rdata1_r[14]),.out(adder_out[14]));
Adder_8bit A15(.in0(adder_in0[15]),.in1(sram_act_rdata1_r[15]),.out(adder_out[15]));


//Layernorm protocol
always @(*) begin//signal
    layernorm_data_in_valid = (current_state == Layernorm_TX) ? 1 : 0;
    layernorm_data_out_ready = (current_state == Layernorm_RX) ? 1 : 0;
end

always @(*) begin//data
    if(layernorm_data_in_valid) begin
        for (i = 0; i < 32; i = i + 1) begin
            layernorm_in_data[i*8 +: 8] = layernorm_buffer0[i];//ex:layernorm_in_data[7:0] = layernorm_buffer0
            layernorm_weights[i*8 +: 8] = layernorm_buffer1[i];
            layernorm_bias[i*8 +: 8]    = layernorm_buffer2[i];
        end
    end
    else begin
        layernorm_in_data = 0;
        layernorm_weights = 0;
        layernorm_bias    = 0;
    end
end

always @(*) begin
    if(layernorm_data_in_valid) begin
        layernorm_in_scale = scale0_model;
        layernorm_weight_scale = scale1_model;
        layernorm_bias_scale = scale2_model;
        layernorm_out_scale = scale3_model;
    end
    else begin
        layernorm_in_scale = 0;
        layernorm_weight_scale = 0;
        layernorm_bias_scale = 0;
        layernorm_out_scale = 0;        
    end
end


endmodule

module pe (

    input clk,
    input rst_n,
    input en,

    input signed [7:0] in0,
    input signed [7:0] in1,
    input signed [7:0] in2,
    input signed [7:0] in3,
    input signed [7:0] in4,
    input signed [7:0] in5,
    input signed [7:0] in6,
    input signed [7:0] in7,
    input signed [7:0] in8,
    input signed [7:0] in9,
    input signed [7:0] in10,
    input signed [7:0] in11,
    input signed [7:0] in12,
    input signed [7:0] in13,
    input signed [7:0] in14,
    input signed [7:0] in15,
    input signed [7:0] in16,
    input signed [7:0] in17,
    input signed [7:0] in18,
    input signed [7:0] in19,
    input signed [7:0] in20,
    input signed [7:0] in21,
    input signed [7:0] in22,
    input signed [7:0] in23,
    input signed [7:0] in24,
    input signed [7:0] in25,
    input signed [7:0] in26,
    input signed [7:0] in27,
    input signed [7:0] in28,
    input signed [7:0] in29,
    input signed [7:0] in30,
    input signed [7:0] in31,

    input signed [7:0] weight0,
    input signed [7:0] weight1,
    input signed [7:0] weight2,
    input signed [7:0] weight3,
    input signed [7:0] weight4,
    input signed [7:0] weight5,
    input signed [7:0] weight6,
    input signed [7:0] weight7,
    input signed [7:0] weight8,
    input signed [7:0] weight9,
    input signed [7:0] weight10,
    input signed [7:0] weight11,
    input signed [7:0] weight12,
    input signed [7:0] weight13,
    input signed [7:0] weight14,
    input signed [7:0] weight15,
    input signed [7:0] weight16,
    input signed [7:0] weight17,
    input signed [7:0] weight18,
    input signed [7:0] weight19,
    input signed [7:0] weight20,
    input signed [7:0] weight21,
    input signed [7:0] weight22,
    input signed [7:0] weight23,
    input signed [7:0] weight24,
    input signed [7:0] weight25,
    input signed [7:0] weight26,
    input signed [7:0] weight27,
    input signed [7:0] weight28,
    input signed [7:0] weight29,
    input signed [7:0] weight30,
    input signed [7:0] weight31,

    output reg signed [24:0]  out
);

    wire signed [15:0] mu0;
    wire signed [15:0] mu1;
    wire signed [15:0] mu2;
    wire signed [15:0] mu3;
    wire signed [15:0] mu4;
    wire signed [15:0] mu5;
    wire signed [15:0] mu6;
    wire signed [15:0] mu7;
    wire signed [15:0] mu8;
    wire signed [15:0] mu9;
    wire signed [15:0] mu10;
    wire signed [15:0] mu11;
    wire signed [15:0] mu12;
    wire signed [15:0] mu13;
    wire signed [15:0] mu14;
    wire signed [15:0] mu15;
    wire signed [15:0] mu16;
    wire signed [15:0] mu17;
    wire signed [15:0] mu18;
    wire signed [15:0] mu19;
    wire signed [15:0] mu20;
    wire signed [15:0] mu21;
    wire signed [15:0] mu22;
    wire signed [15:0] mu23;
    wire signed [15:0] mu24;
    wire signed [15:0] mu25;
    wire signed [15:0] mu26;
    wire signed [15:0] mu27;
    wire signed [15:0] mu28;
    wire signed [15:0] mu29;
    wire signed [15:0] mu30;
    wire signed [15:0] mu31;

    assign mu0 = in0 * weight0;
    assign mu1 = in1 * weight1;
    assign mu2 = in2 * weight2;
    assign mu3 = in3 * weight3;
    assign mu4 = in4 * weight4;
    assign mu5 = in5 * weight5;
    assign mu6 = in6 * weight6;
    assign mu7 = in7 * weight7;
    assign mu8 = in8 * weight8;
    assign mu9 = in9 * weight9;
    assign mu10 = in10 * weight10;
    assign mu11 = in11 * weight11;
    assign mu12 = in12 * weight12;
    assign mu13 = in13 * weight13;
    assign mu14 = in14 * weight14;
    assign mu15 = in15 * weight15;
    assign mu16 = in16 * weight16;
    assign mu17 = in17 * weight17;
    assign mu18 = in18 * weight18;
    assign mu19 = in19 * weight19;
    assign mu20 = in20 * weight20;
    assign mu21 = in21 * weight21;
    assign mu22 = in22 * weight22;
    assign mu23 = in23 * weight23;
    assign mu24 = in24 * weight24;
    assign mu25 = in25 * weight25;
    assign mu26 = in26 * weight26;
    assign mu27 = in27 * weight27;
    assign mu28 = in28 * weight28;
    assign mu29 = in29 * weight29;
    assign mu30 = in30 * weight30;
    assign mu31 = in31 * weight31;

always@(posedge clk)begin
    if(!rst_n) out <= 0;
    else if (!en) out <= 0;
    else begin
        out <= out +
       (((((mu0) + (mu1)) + ((mu2) + (mu3))) +
        (((mu4) + (mu5)) + ((mu6) + (mu7)))) +
       ((((mu8) + (mu9)) + ((mu10) + (mu11))) +
        (((mu12) + (mu13)) + ((mu14) + (mu15))))) +
      (((((mu16) + (mu17)) + ((mu18) + (mu19))) +
        (((mu20) + (mu21)) + ((mu22) + (mu23)))) +
       ((((mu24) + (mu25)) + ((mu26) + (mu27))) +
        (((mu28) + (mu29)) + ((mu30) + (mu31)))));
    end
end


endmodule

module requantization (
    input signed [24:0] in_data,
    input signed [24:0] scale,
    output reg signed  [7:0] out_data
);
    wire signed [26:0] temp;
    wire signed [10:0] data;
    wire signed [7:0] data1;

    assign temp = in_data * scale;
    assign data = (temp >>> 16);
    assign data1 = (data > 127) ? 127 : (data < -128) ? -128 : data;

    always @(*) begin
        out_data = data1;
    end

endmodule

module Adder_8bit(
    input  signed [7:0] in0,
    input  signed [7:0] in1,
    output signed [8:0] out    
);

assign out = in0 + in1;

endmodule