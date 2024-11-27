# add your source code
SIM_SRC = sim_rtl.f
SYN_SRC = sim_gate.f
BAK = *.bak
LOG = *.log *.history *.key *.fsdb out_log.txt novas.rc novas.conf *.sdf.X
INCA_libs = INCA_libs
cad_dir = spyglass-* nWaveLog* BSSLib.lib*

all :: sim

sim:
	vcs   -full64 -R -debug_access+all +v2k +neg_tchk -notice -Mupdate -cm line+cond+fsm+tgl -f ${SIM_SRC} | tee sim.log

wave:
	nWave -ssr signal.rc &


clean:
	-rm -f ${BAK} ${LOG}
	-rm -rf ${INCA_libs} ${cad_dir}
