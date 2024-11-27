# Testbench
bert_encoder_tb.v
# SRAM behavior model
sram_model/SRAM_activation_4096x128b.v
sram_model/SRAM_weight_16384x128b.v
# non linear model
non_linear_model/softmax.v
non_linear_model/layernorm.v
non_linear_model/GELU.v
# Add your design here
../../hdl/bert_encoder.v

+access+r
