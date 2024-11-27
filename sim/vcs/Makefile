# add your source code
SIM_SRC = sim_rtl.f
SYN_SRC = sim_gate.f
BAK = *.bak
LOG = *.log *.history *.key *.fsdb out_log.txt novas.rc novas.conf *.sdf.X
INCA_libs = INCA_libs
cad_dir = spyglass-* nWaveLog* BSSLib.lib*

all :: sim

sim:
	vcs   -full64 -R -debug_access+all +v2k +neg_tchk  -f ${SIM_SRC} | tee rtl_sim.log

wave:
	nWave -ssr signal.rc &

syn:
	vcs  -full64 -R -debug_access+all +v2k +neg_tchk -notice -Mupdate -cm line+cond+fsm+tgl -f ${SYN_SRC} | tee gate_sim.log

clean:
	-rm -f ${BAK} ${LOG}
	-rm -rf ${INCA_libs} ${cad_dir}
