module GELU( 
    input wire clk,
    input wire rst,
    input wire data_in_valid,
    input wire data_out_ready,
    input wire [255:0] in_data,
    input wire [31:0] in_scale,
    input wire [31:0] out_scale,
    output data_out_valid,
    output data_in_ready,
    output reg [255:0] out_data
);
    real in_data_real[31:0];
    real out_data_real[31:0];
    integer i;
    real scale;
    reg [2:0] state, state_n;

    
    always @(posedge clk, negedge rst) begin
        if (~rst) begin
            out_data <= 0;
            for (i=0 ; i<32 ; i=i+1) 
                in_data_real[i] <= 0;
        end
        else begin
            if (data_in_valid) begin
                for (i=0 ; i< 32 ; i=i+1) 
                    in_data_real[i] <= $signed(in_data[8*i+7 -: 8])* (in_scale  / $pow(2,16));
            end
            for (i=0 ; i<32 ; i=i+1) 
                out_data[8*i+7 -: 8] <= out_data_real[i] > 127 ? 127 : out_data_real[i] < -128 ? -128 : out_data_real[i];
        end
    end

     always @(*) begin
        case(state)
            0: begin
                for (i=0 ; i<32 ; i=i+1)
                    out_data_real[i] = 0;
                if (data_in_valid) 
                    state_n = 1;
                else 
                    state_n = 0;
            end
            1: begin
                for ( i=0 ; i< 32; i=i+1)
                    out_data_real[i] = $floor((0.5 * in_data_real[i] * (1 + $tanh(0.79788 * (in_data_real[i]  + 0.044715 * $pow(in_data_real[i], 3))))) * $pow(2,16) / out_scale) ;
                state_n = 2;
            end
            2: begin
                for ( i=0 ; i<32 ; i=i+1) begin
                    out_data_real[i] = $signed(out_data[8*i+7 -: 8]);
                end
                if (data_out_ready) 
                    state_n = 0;
                else 
                    state_n = 2;    
            end
            default: state_n = 0;
        endcase     
    end
    
    always @(posedge clk) begin
        state <= state_n;
    end

    assign data_out_valid = state == 2;
    assign data_in_ready = state == 0;



endmodule
