@echo off
echo 🔪 Stopping Distributed LLM Cluster (killing by window title)...

:: Kill each component by the window title assigned in run_system.bat
taskkill /FI "WINDOWTITLE eq Scheduler*" /F /T >nul 2>&1
taskkill /FI "WINDOWTITLE eq Worker 8000*" /F /T >nul 2>&1
taskkill /FI "WINDOWTITLE eq Worker 8001*" /F /T >nul 2>&1
taskkill /FI "WINDOWTITLE eq Worker 8002*" /F /T >nul 2>&1
taskkill /FI "WINDOWTITLE eq Worker 8003*" /F /T >nul 2>&1

:: Also free the ports using netstat in case windows linger
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":9000 "') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000 "') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8001 "') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8002 "') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8003 "') do taskkill /PID %%a /F >nul 2>&1

echo ✅ Done. All cluster processes cleared.
pause
