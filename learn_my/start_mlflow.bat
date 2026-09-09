@echo off
setlocal

set "QLIB_ROOT=C:\Users\WANGKANG\Desktop\qlib"
set "MLFLOW_EXE=C:\Users\WANGKANG\miniconda3\envs\qlib\Scripts\mlflow.exe"
set "MLFLOW_STORE=%QLIB_ROOT%\mlruns"
set "MLFLOW_ARTIFACTS=%MLFLOW_STORE%\artifacts"
set "MLFLOW_DB_URI=sqlite:///C:/Users/WANGKANG/Desktop/qlib/mlruns/mlflow.db"
set "MLFLOW_ARTIFACT_URI=file:///C:/Users/WANGKANG/Desktop/qlib/mlruns/artifacts"

if not exist "%MLFLOW_EXE%" (
    echo [ERROR] Qlib MLflow was not found:
    echo         %MLFLOW_EXE%
    pause
    exit /b 1
)

if not exist "%MLFLOW_STORE%" mkdir "%MLFLOW_STORE%"
if not exist "%MLFLOW_ARTIFACTS%" mkdir "%MLFLOW_ARTIFACTS%"

powershell.exe -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 (
    echo [ERROR] Port 5000 is already in use. MLflow may already be running.
    pause
    exit /b 1
)

cd /d "%QLIB_ROOT%"

echo Starting MLflow from the qlib Conda environment...
echo UI: http://127.0.0.1:5000
echo Database: %MLFLOW_STORE%\mlflow.db
echo Artifacts: %MLFLOW_ARTIFACTS%
echo.

"%MLFLOW_EXE%" server ^
  --backend-store-uri "%MLFLOW_DB_URI%" ^
  --artifacts-destination "%MLFLOW_ARTIFACT_URI%" ^
  --host 127.0.0.1 ^
  --port 5000 ^
  --workers 1

echo.
echo MLflow Server stopped with exit code %ERRORLEVEL%.
pause
endlocal
