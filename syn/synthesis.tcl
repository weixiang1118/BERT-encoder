set cycle 10
analyze -format verilog  { ../hdl/bert_encoder.v  }

elaborate bert_encoder -architecture verilog 

source compile.tcl
