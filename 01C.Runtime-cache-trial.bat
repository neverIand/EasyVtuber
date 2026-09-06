@echo off
setlocal
cd /D "%~dp0"

set "PATH=%~dp0envs\TensorRT-RTX-1.3.0.35_cu129\bin;%~dp0envs\python_embedded;%~dp0envs\python_embedded\Scripts;%~dp0envs\python_embedded\Library\bin;%PATH%"
if not defined EZVTB_TRT_RUNTIME_CACHE set "EZVTB_TRT_RUNTIME_CACHE=1"
set "PYTHONUNBUFFERED=1"

echo Runtime cache is now enabled by default in launchers A and B.
echo Use launcher B for everyday debug output. This legacy entry uses the same settings.
"%~dp0envs\python_embedded\python.exe" -u launcher2.py

pause
endlocal
