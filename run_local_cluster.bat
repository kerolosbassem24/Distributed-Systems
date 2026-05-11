@echo off
echo Starting Distributed LLM System Local Cluster...

:: Start workers in the background
echo Starting Worker 1 on port 8001...
start "Worker 8001" cmd /k python workers/worker_server.py --port 8001 --gpu-id 0 --mock

echo Starting Worker 2 on port 8002...
start "Worker 8002" cmd /k python workers/worker_server.py --port 8002 --gpu-id 0 --mock

echo Starting Worker 3 on port 8003...
start "Worker 8003" cmd /k python workers/worker_server.py --port 8003 --gpu-id 0 --mock

echo Starting Worker 4 on port 8004...
start "Worker 8004" cmd /k python workers/worker_server.py --port 8004 --gpu-id 0 --mock
:: Wait a few seconds for servers to start
echo Waiting for workers to initialize...
timeout /t 5 /nobreak > nul

:: Start main load balancer and scheduler
echo Starting Gateway / Load Balancer (main.py)...
python main.py

echo.
echo Cluster execution finished. To kill background workers, you can run:
echo taskkill /F /IM python.exe
