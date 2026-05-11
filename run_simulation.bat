@echo off
setlocal EnableDelayedExpansion

title Distributed LLM — User Simulation Benchmark
color 0B

echo.
echo  ============================================================
echo   DISTRIBUTED LLM SYSTEM — USER SIMULATION BENCHMARK
echo   Scales: 100 / 250 / 500 / 1000 concurrent users
echo  ============================================================
echo.

:: ── locate venv ──────────────────────────────────────────────────────────────
set "SCRIPT_DIR=%~dp0"
set "VENV_PYTHON=%SCRIPT_DIR%venv\Scripts\python.exe"
set "SIM_SCRIPT=%SCRIPT_DIR%client\user_simulation.py"

if not exist "%VENV_PYTHON%" (
    echo  [ERROR] Virtual environment not found at:
    echo          %VENV_PYTHON%
    echo.
    echo  Please create and activate the venv first:
    echo      python -m venv venv
    echo      venv\Scripts\activate
    echo      pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: ── ensure httpx is installed ────────────────────────────────────────────────
echo  Checking dependencies...
"%VENV_PYTHON%" -c "import httpx" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo  Installing httpx...
    "%VENV_PYTHON%" -m pip install httpx --quiet
)

"%VENV_PYTHON%" -c "import colorama" 2>NUL
if %ERRORLEVEL% NEQ 0 (
    echo  Installing colorama (optional, for colored output)...
    "%VENV_PYTHON%" -m pip install colorama --quiet
)

echo.
echo  ── Server Check ─────────────────────────────────────────────
echo  Make sure app.py is running BEFORE continuing.
echo  Start it in a separate terminal with:
echo.
echo      cd /d "%SCRIPT_DIR%"
echo      venv\Scripts\activate
echo      python app.py
echo.
echo  Press any key when the server is ready, or Ctrl+C to abort.
pause >NUL

echo.
echo  ── Starting Simulation ──────────────────────────────────────
echo.

:: ── Run with default scales (100 250 500 1000) ───────────────────────────────
:: To override, edit --scales below or add arguments
"%VENV_PYTHON%" "%SIM_SCRIPT%" ^
    --url     http://localhost:8000 ^
    --scales  100 250 500 1000 ^
    --timeout 120 ^
    --workers 64 ^
    --out     "%SCRIPT_DIR%results"

echo.
echo  ── Done ─────────────────────────────────────────────────────
echo  Results saved in:  %SCRIPT_DIR%results\
echo.
pause
