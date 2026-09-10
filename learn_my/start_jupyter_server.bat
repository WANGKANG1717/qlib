@echo off
setlocal

set "QLIB_ROOT=C:\Users\WANGKANG\Desktop\qlib"
set "JUPYTER_TOKEN=token"

"C:\Users\WANGKANG\miniconda3\envs\qlib\Scripts\jupyter-server.exe" ^
  --ServerApp.root_dir="%QLIB_ROOT%" ^
  --IdentityProvider.token="%JUPYTER_TOKEN%" ^
  --ServerApp.ip=127.0.0.1 ^
  --port=8888 ^
  --no-browser

pause
