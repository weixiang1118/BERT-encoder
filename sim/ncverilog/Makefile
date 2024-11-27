# add your source code
SIM_SRC = sim_rtl.f
SYN_SRC = sim_gate.f
BAK = *.bak
LOG = *.log *.history *.key *.fsdb out_log.txt novas.rc novas.conf *.sdf.X
INCA_libs = INCA_libs
cad_dir = spyglass-* nWaveLog* BSSLib.lib*

all :: sim

sim:
	ncverilog -f ${SIM_SRC} | tee sim.log && vcd2fsdb bert_encoder.vcd -o bert_encoder.fsdb
syn:
	ncverilog -f ${SYN_SRC} | tee gate_sim.log && vcd2fsdb bert_encoder_syn.vcd -o bert_encoder_syn.fsdb

clean:
	-rm -f ${BAK} ${LOG}
	-rm -rf ${INCA_libs} ${cad_dir}
