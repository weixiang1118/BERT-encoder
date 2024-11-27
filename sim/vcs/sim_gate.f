# testbench
bert_encoder_tb.v
# SRAM behavior model
sram_model/SRAM_activation_4096x128b.v
sram_model/SRAM_weight_16384x128b.v
# non linear model
non_linear_model/softmax.v
non_linear_model/layernorm.v
non_linear_model/GELU.v

# netlist generated by Design Compiler
.../syn/netlist/bert_encoder_syn.v
# Logic Gate models
-v  /home/m110/m110061576/process/CBDK_TSMC90GUTM_Arm_f1.0/orig_lib/aci/sc-x/verilog/tsmc090.v

+define+GATESIM
#+access+r
