module SRAM_weight_16384x128b( 
    input wire clk,
    input wire [15:0] wea0,
    input wire [15:0] addr0,
    input wire [127:0] wdata0,
    output reg [127:0] rdata0,
    input wire [15:0] wea1,
    input wire [15:0] addr1,
    input wire [127:0] wdata1,
    output reg [127:0] rdata1
);

    reg [127:0] RAM [0:16383];
    wire [127:0] masked_wdata0, masked_wdata1;
    
    assign masked_wdata0[ 7: 0] = (wea0[0] ? wdata0[ 7: 0] : RAM[addr0][ 7: 0]);
    assign masked_wdata0[15: 8] = (wea0[1] ? wdata0[15: 8] : RAM[addr0][15: 8]);
    assign masked_wdata0[23:16] = (wea0[2] ? wdata0[23:16] : RAM[addr0][23:16]);
    assign masked_wdata0[31:24] = (wea0[3] ? wdata0[31:24] : RAM[addr0][31:24]);
    assign masked_wdata0[39:32] = (wea0[4] ? wdata0[39:32] : RAM[addr0][39:32]);
    assign masked_wdata0[47:40] = (wea0[5] ? wdata0[47:40] : RAM[addr0][47:40]);
    assign masked_wdata0[55:48] = (wea0[6] ? wdata0[55:48] : RAM[addr0][55:48]);
    assign masked_wdata0[63:56] = (wea0[7] ? wdata0[63:56] : RAM[addr0][63:56]);
    assign masked_wdata0[71:64] = (wea0[8] ? wdata0[71:64] : RAM[addr0][71:64]);
    assign masked_wdata0[79:72] = (wea0[9] ? wdata0[79:72] : RAM[addr0][79:72]);
    assign masked_wdata0[87:80] = (wea0[10] ? wdata0[87:80] : RAM[addr0][87:80]);
    assign masked_wdata0[95:88] = (wea0[11] ? wdata0[95:88] : RAM[addr0][95:88]);
    assign masked_wdata0[103:96] = (wea0[12] ? wdata0[103:96] : RAM[addr0][103:96]);
    assign masked_wdata0[111:104] = (wea0[13] ? wdata0[111:104] : RAM[addr0][111:104]);
    assign masked_wdata0[119:112] = (wea0[14] ? wdata0[119:112] : RAM[addr0][119:112]);
    assign masked_wdata0[127:120] = (wea0[15] ? wdata0[127:120] : RAM[addr0][127:120]);

    assign masked_wdata1[ 7: 0] = (wea1[0] ? wdata1[ 7: 0] : RAM[addr1][ 7: 0]);
    assign masked_wdata1[15: 8] = (wea1[1] ? wdata1[15: 8] : RAM[addr1][15: 8]);
    assign masked_wdata1[23:16] = (wea1[2] ? wdata1[23:16] : RAM[addr1][23:16]);
    assign masked_wdata1[31:24] = (wea1[3] ? wdata1[31:24] : RAM[addr1][31:24]);
    assign masked_wdata1[39:32] = (wea1[4] ? wdata1[39:32] : RAM[addr1][39:32]);
    assign masked_wdata1[47:40] = (wea1[5] ? wdata1[47:40] : RAM[addr1][47:40]);
    assign masked_wdata1[55:48] = (wea1[6] ? wdata1[55:48] : RAM[addr1][55:48]);
    assign masked_wdata1[63:56] = (wea1[7] ? wdata1[63:56] : RAM[addr1][63:56]);
    assign masked_wdata1[71:64] = (wea1[8] ? wdata1[71:64] : RAM[addr1][71:64]);
    assign masked_wdata1[79:72] = (wea1[9] ? wdata1[79:72] : RAM[addr1][79:72]);
    assign masked_wdata1[87:80] = (wea1[10] ? wdata1[87:80] : RAM[addr1][87:80]);
    assign masked_wdata1[95:88] = (wea1[11] ? wdata1[95:88] : RAM[addr1][95:88]);
    assign masked_wdata1[103:96] = (wea1[12] ? wdata1[103:96] : RAM[addr1][103:96]);
    assign masked_wdata1[111:104] = (wea1[13] ? wdata1[111:104] : RAM[addr1][111:104]);
    assign masked_wdata1[119:112] = (wea1[14] ? wdata1[119:112] : RAM[addr1][119:112]);
    assign masked_wdata1[127:120] = (wea1[15] ? wdata1[127:120] : RAM[addr1][127:120]);

    always @(posedge clk) begin
        RAM[addr0] <= #(`CYCLE*0.5)masked_wdata0;
        if(addr0 != addr1)
            RAM[addr1] <= #(`CYCLE*0.5)masked_wdata1;
    end 

    always @(posedge clk) begin
        rdata0 <= #(`CYCLE*0.5)RAM[addr0];
    end
    always @(posedge clk) begin
        rdata1 <= #(`CYCLE*0.5)RAM[addr1];
    end
    
    
    task load_data(
        input [511:0] file_name
    );
        $readmemh(file_name, RAM);
    endtask

endmodule
