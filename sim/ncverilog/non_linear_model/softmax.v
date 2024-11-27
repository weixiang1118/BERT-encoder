module softmax ( 
    input wire clk,
    input wire rst,
    input wire data_in_valid,
    input wire data_out_ready,
    input wire [255:0] in_data,
    input wire [31:0] in_scale,
    input wire [31:0] out_scale,
    input wire [7:0] S,
    output reg data_out_valid,
    output reg data_in_ready,    
    output reg [255:0] out_data
);
    real in_buffer [0:31]; 
    real exp_data [0:31], exp_data_n[0:31];
    real sum_data, sum_data_n;
    real div_data, div_data_n;
    real in_data_real, in_data_real_n;
    real out_data_real[0:31];
    parameter negative_value = -1000;
    reg [2:0] state, state_n;
    integer i;
    real scale;
    
    always @(posedge clk or negedge rst) begin
        if (~rst) begin
            state <= 0;
            out_data <= 0;
            for (i=0 ; i<32 ; i=i+1) begin
                in_buffer[i] <= 0;
            end
        end
        else begin
            if (data_in_valid) begin
                for (i=0 ; i<32 ; i=i+1) 
                    in_buffer[i] <= $signed(in_data[8*i+7 -: 8]);
            end
            for (i=0 ; i<32 ; i=i+1) 
                out_data[8*i+7 -: 8] <= out_data_real[i] > 127 ? 127 : out_data_real[i] < -128 ? -128 : out_data_real[i];
            state <= state_n;
        end 
    end

    always @(*) begin
        case(state)
            0: begin
                sum_data = 0;
                for (i=0 ; i<32 ; i=i+1)begin
                    exp_data[i] = 0;
                    out_data_real[i] = 0;
                end
                if (data_in_valid) 
                    state_n = 1;
                else 
                    state_n = 0;
            end
            1: begin
                sum_data = 0;
                for ( i=0 ; i< 32; i=i+1) begin
                    if (i < S) 
                        exp_data[i] = $exp((in_buffer[i] * in_scale) / $pow(2,16));
                    else 
                        exp_data[i] = $exp(negative_value);
                    sum_data = sum_data + exp_data[i];
                end
                for ( i=0 ; i<32 ; i=i+1) 
                    out_data_real[i] = $floor(exp_data[i] * $pow(2,16) / sum_data / out_scale) ;
                state_n = 2;
            end
            2: begin
                sum_data = 0;
                for ( i=0 ; i<32 ; i=i+1)begin
                    exp_data[i] = 0;
                    out_data_real[i] = out_data[8*i+7 -: 8];
                end
                if (data_out_ready) 
                    state_n = 0;
                else 
                    state_n = 2;    
            end
            default: state_n = 0;
        endcase     
    end

    always @(*) begin
        if (state == 2)
            data_out_valid = 1;
        else
            data_out_valid = 0;
        if (state == 0)
            data_in_ready = 1;
        else
            data_in_ready = 0;
    end

endmodule
