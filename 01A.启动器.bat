@echo on
setlocal
cd /D "%~dp0"

SET PATH=%~dp0envs\TensorRT-RTX-1.3.0.35_cu129\bin;%~dp0envs\python_embedded;%~dp0envs\python_embedded\Scripts;%~dp0envs\python_embedded\Library\bin;%PATH%

if not defined EZVTB_TRT_RUNTIME_CACHE set "EZVTB_TRT_RUNTIME_CACHE=1"
set "PYTHONUNBUFFERED=1"

start "" "%~dp0envs\python_embedded\pythonw.exe" launcher2.py
endlocal
exit /b
