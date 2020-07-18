@echo off

set DARKNET_DIR=C:\darknet

set START_TIME=%DATE% %TIME%

powershell -command "%DARKNET_DIR%\darknet.exe detector train datasets.data yolov4-custom.cfg 2>&1 | add-content -path log.txt -passthru"

set END_TIME=%DATE% %TIME%

echo "START = "%START_TIME%
echo "END   = "%END_TIME%

pause
