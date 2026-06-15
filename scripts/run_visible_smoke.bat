@echo off
chcp 65001>nul
setlocal

set "LAB_DIR=%~dp0.."
set "D2S_DIR=%LAB_DIR%\..\Desktop2Stereo_v2.4.2_Windows_NVIDIA\Desktop2Stereo"
set "PYTHON_EXE=%D2S_DIR%\python3\python.exe"

if not exist "%PYTHON_EXE%" (
  echo [Error] Python not found: %PYTHON_EXE%
  pause
  exit /b 1
)

title 4K Stereo Lab - Smoke Test
echo [Info] Using Desktop2Stereo Python:
echo        %PYTHON_EXE%
echo [Info] First torch/CUDA import may take several minutes on low-end machines.
echo.

pushd "%D2S_DIR%"
"%PYTHON_EXE%" "%LAB_DIR%\scripts\smoke_test.py"
set "EXIT_CODE=%ERRORLEVEL%"
popd

echo.
echo [Info] Exit code: %EXIT_CODE%
pause
exit /b %EXIT_CODE%
