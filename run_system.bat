@echo off
TITLE LLM Distributed System Manager
echo ====================================================
echo   🚀 STARTING DISTRIBUTED LLM CLUSTER 🚀
echo   (Configured for Single GPU / Mock Mode)
echo ====================================================

:: Use the virtual environment python
set PYTHON_EXE=venv\Scripts\python.exe

if not exist %PYTHON_EXE% (
    echo [ERROR] Virtual environment not found at .\venv
    pause
    exit /b
)

:: Since you have 1 GPU, we will run them on GPU 0 or in Mock mode.
:: Change --gpu-id 0 to --mock if you run out of memory.
set WORKER_FLAGS=--gpu-id 0

echo [1/5] Starting Scheduler...
start "Scheduler" cmd /k "%PYTHON_EXE% scheduler.py"
timeout /t 3 /nobreak > nul

echo [2/5] Starting Worker 1 (Port 8000)...
start "Worker 8000" cmd /k "%PYTHON_EXE% gpu_worker_server.py --port 8000 %WORKER_FLAGS% --worker-id thunder-gpu-1"
timeout /t 1 /nobreak > nul

echo [3/5] Starting Worker 2 (Port 8001)...
start "Worker 8001" cmd /k "%PYTHON_EXE% gpu_worker_server.py --port 8001 %WORKER_FLAGS% --worker-id thunder-gpu-2"
timeout /t 1 /nobreak > nul

echo [4/5] Starting Worker 3 (Port 8002)...
start "Worker 8002" cmd /k "%PYTHON_EXE% gpu_worker_server.py --port 8002 %WORKER_FLAGS% --worker-id thunder-gpu-3"
timeout /t 1 /nobreak > nul

echo [5/5] Starting Worker 4 (Port 8003)...
start "Worker 8003" cmd /k "%PYTHON_EXE% gpu_worker_server.py --port 8003 %WORKER_FLAGS% --worker-id thunder-gpu-4"

echo.
echo ====================================================
echo   ✅ CLUSTER STARTUP INITIATED
echo   I have used 'cmd /k' so the windows stay open
echo   if a worker crashes. Check them for errors!
echo ====================================================
pause
