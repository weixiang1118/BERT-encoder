wvRestoreSignal -win $_nWave1 "signal.rc" -overWriteAutoAlias on -appendSignals \
           on
wvResizeWindow -win $_nWave1 8 31 892 72
wvResizeWindow -win $_nWave1 0 23 1920 1009
wvResizeWindow -win $_nWave1 8 31 892 72
wvResizeWindow -win $_nWave1 0 23 1920 1009
wvResizeWindow -win $_nWave1 0 23 1920 1009
wvResizeWindow -win $_nWave1 0 23 1920 1009
wvResizeWindow -win $_nWave1 0 23 1920 1009
wvResizeWindow -win $_nWave1 1715 69 1920 1057
wvResizeWindow -win $_nWave1 1920 23 1920 1009
wvSelectSignal -win $_nWave1 {( "layernorm" 13 )} 
wvSelectSignal -win $_nWave1 {( "layernorm" 12 )} 
wvSelectSignal -win $_nWave1 {( "layernorm" 7 )} 
wvSelectSignal -win $_nWave1 {( "layernorm" 7 )} 
wvSelectSignal -win $_nWave1 {( "layernorm" 8 )} 
wvSelectSignal -win $_nWave1 {( "layernorm" 9 )} 
wvSelectSignal -win $_nWave1 {( "Softmax" 5 )} 
wvSetCursor -win $_nWave1 58272.901272 -snap {("layernorm" 9)}
wvSelectSignal -win $_nWave1 {( "layernorm" 2 )} 
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSetCursor -win $_nWave1 11099.600242 -snap {("layernorm" 2)}
wvSelectSignal -win $_nWave1 {( "layernorm" 7 )} 
wvSetCursor -win $_nWave1 59858.558449 -snap {("layernorm" 9)}
wvSetCursor -win $_nWave1 75715.130224 -snap {("layernorm" 1)}
wvSetCursor -win $_nWave1 90382.459116 -snap {("layernorm" 1)}
wvSetCursor -win $_nWave1 101878.473652 -snap {("layernorm" 1)}
wvSetCursor -win $_nWave1 95535.844942 -snap {("layernorm" 7)}
wvSetCursor -win $_nWave1 100689.230769 -snap {("layernorm" 8)}
wvSetCursor -win $_nWave1 60254.972744 -snap {("layernorm" 9)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSelectSignal -win $_nWave1 {( "layernorm" 9 )} 
wvSetCursor -win $_nWave1 65804.772865 -snap {("layernorm" 7)}
wvSetCursor -win $_nWave1 61047.801333 -snap {("layernorm" 2)}
wvSetCursor -win $_nWave1 65011.944276 -snap {("layernorm" 7)}
wvSetCursor -win $_nWave1 60651.387038 -snap {("layernorm" 2)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSetCursor -win $_nWave1 65804.772865 -snap {("layernorm" 9)}
wvSetCursor -win $_nWave1 78490.030285 -snap {("layernorm" 7)}
wvSetCursor -win $_nWave1 90778.873410 -snap {("layernorm" 7)}
wvSetCursor -win $_nWave1 103067.716535 -snap {("layernorm" 7)}
wvSetCursor -win $_nWave1 112978.073895 -snap {("layernorm" 8)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSelectSignal -win $_nWave1 {( "layernorm" 10 )} 
wvSelectSignal -win $_nWave1 {( "layernorm" 9 )} 
wvSetCursor -win $_nWave1 68183.258631 -snap {("layernorm" 9)}
wvSelectStuckSignals -win $_nWave1
wvSelectGroup -win $_nWave1 {G3}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvSetCursor -win $_nWave1 76507.958813 -snap {("layernorm" 7)}
wvSetCursor -win $_nWave1 90778.873410 -snap {("layernorm" 10)}
wvDisplayGridCount -win $_nWave1 -off
wvGetSignalClose -win $_nWave1
wvReloadFile -win $_nWave1
wvExit
