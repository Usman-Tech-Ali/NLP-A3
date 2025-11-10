@echo off
REM Stock Forecasting Application Startup Script for Windows

echo Starting Stock Forecasting Application...

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo Starting Flask application...
python backend\app.py

pause
