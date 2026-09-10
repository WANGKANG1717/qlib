@echo off
setlocal

cd /d "%~dp0"

set "PYTHON_EXE=C:\Users\WANGKANG\miniconda3\envs\qlib\python.exe"
set "MLRUNS_DIR=C:\Users\WANGKANG\Desktop\qlib\mlruns"
set "MLFLOW_URL=http://127.0.0.1:5000"
set "MLFLOW_ALLOW_FILE_STORE=true"

if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python was not found:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)

if not exist "%MLRUNS_DIR%" (
    echo [ERROR] MLflow data directory was not found:
    echo %MLRUNS_DIR%
    pause
    exit /b 1
)

echo Starting MLflow UI...
echo Data: %MLRUNS_DIR%
echo URL:  %MLFLOW_URL%
echo Close the "MLflow UI" window to stop the server.

"%PYTHON_EXE%" -m mlflow ui --backend-store-uri "file://%MLRUNS_DIR%" --host 127.0.0.1 --port 5000
timeout /t 2 /nobreak >nul
start "" "%MLFLOW_URL%"

