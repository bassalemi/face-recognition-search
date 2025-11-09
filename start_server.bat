@echo off
REM GPU-Accelerated Face Recognition Server Launcher
REM Sets up CUDA paths and starts the Flask server

echo ========================================
echo Starting Face Recognition Server...
echo ========================================

REM Set CUDA paths for GPU acceleration
set PATH=C:\Program Files\NVIDIA\CUDNN\v9.15\bin\12.9;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin;%PATH%

REM Navigate to project directory
cd /d "d:\Python codes\Face recognition"

REM Run the server with virtual environment Python
.\venv\Scripts\python.exe app.py

pause
