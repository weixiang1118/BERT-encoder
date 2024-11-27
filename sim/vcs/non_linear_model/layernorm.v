
module layernorm( 
    input wire clk,
    input wire rst,
    input wire data_in_valid,
    input wire data_out_ready,
    input wire [255:0] in_data,
    input wire [255:0] weights,
    input wire [255:0] bias,
    input wire [31:0] in_scale,
    input wire [31:0] weight_scale,
    input wire [31:0] bias_scale,
    input wire [31:0] out_scale,
    output  data_out_valid,
    output  data_in_ready,
    output reg [255:0] out_data
);

    real in_data_real[127:0];
    real out_data_real[127:0], out_data_real_n[127:0];
    real weights_real[127:0];
    real bias_real[127:0];
    real scale, out_scale_real, bias_scale_real, weight_scale_real;
    integer i;
    reg [1:0] state, state_n;
    real mean, variance;
    reg [2:0] count_in, count_in_n;
    reg [2:0] count_out, count_out_n;

    always@(posedge clk or negedge rst)begin
        if(~rst)begin
            out_data <= 0;
            out_scale_real <= 0;
            for (i=0; i<128; i=i+1)begin
                in_data_real[i] <= 0;
                weights_real[i] <= 0;
                bias_real[i] <= 0;
                in_data_real[i] <= 0;
            end
        end
        else begin
            if(data_in_valid)begin
                for(i=0; i<32; i=i+1)begin    
                    in_data_real[count_in * 32 + i] <= $signed(in_data[i*8+7 -: 8]) * (in_scale  / $pow(2,16));
                    weights_real[count_in * 32 + i] <= $signed(weights[i*8+7 -: 8]) * (weight_scale / $pow(2,16));
                    bias_real[count_in * 32 + i] <= $signed(bias[i*8+7 -: 8]) * (bias_scale / $pow(2,16));
                end
                out_scale_real <= out_scale / $pow(2,16);
            end
            
            for(i=0; i<32; i=i+1)
                out_data[i*8+7 -: 8] <= out_data_real[count_out_n*32 + i] > 127 ? 127 : out_data_real[count_out_n*32 + i] < -128 ? -128 : $floor(out_data_real[count_out_n*32 + i]);
        
        end
    end

    always@(*)begin
        case(state)
            0:begin
                if(data_in_valid)
                    count_in_n = count_in + 1;
                else
                    count_in_n = 0;
            end
            1:begin
                if(data_in_valid)begin
                    if(count_in == 4)
                        count_in_n = 0;
                    else
                        count_in_n = count_in + 1;
                end
                else
                    count_in_n = count_in;
                
            end
            2:begin
                count_in_n = 0;
            end
            default: count_in_n = count_in;
        endcase
    end

    always@(*)begin
        case(state)
            0, 1, 2:begin
                count_out_n = 0;
            end
            3:begin
                if(data_out_ready)
                    count_out_n = count_out + 1;
                else
                    count_out_n = count_out;
            end
        endcase
    end

    always@(*)begin
        if( state == 2)begin
            mean = 0;
            variance = 0;
            
            for (i=0; i<128; i=i+1)
                mean = mean + in_data_real[i] ;
            mean = mean / 128;
            for(i=0; i<128; i=i+1)
                variance = variance + $pow(in_data_real[i] - mean, 2);
            variance = variance / 128;
            for (i=0 ; i<128; i=i+1)
                out_data_real[i] = $floor(((in_data_real[i]- mean) * weights_real[i] / $sqrt(variance + 0.000001) + bias_real[i] )/ out_scale_real);

        end
        else begin
            mean = 0;
            variance = 0;
            for(i=0; i<128; i=i+1) 
                out_data_real[i] = out_data_real_n[i];
        end 
    end

    always@(*)begin
        case(state)
            default: state_n = 0;
            0:begin
                if(data_in_valid)
                    state_n = 1;
                else
                    state_n = 0;
            end
            1:begin
                if(count_in == 3 && data_in_valid)
                    state_n = 2;
                else
                    state_n = 1;
            end
            2:begin
                state_n = 3;
            end
            3:begin
                if(count_out == 3)begin
                    state_n = 0;
                end
                else
                    state_n = 3;
            end
        endcase

    end

    always@(posedge clk or negedge rst)begin
        if(~rst)begin
            state <= 0;
            count_in <= 0;
            count_out <= 0;
            for(i=0; i<128; i=i+1)
                out_data_real[i] <= 0;
        end
        else begin
            state <= state_n;
            count_in <= count_in_n;
            count_out <= count_out_n;
            for(i=0; i<128; i=i+1)
                out_data_real_n[i] = out_data_real[i];
        end
    end
    
    assign data_out_valid = state == 3;
    assign data_in_ready = state == 1 || state == 0;
endmodule