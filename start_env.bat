@echo off
SET VENV_DIR=venv

REM Check if venv exists, if not, create it
IF NOT EXIST %VENV_DIR% (
    echo Creating virtual environment...
    python -m venv %VENV_DIR%
)

REM Activate the virtual environment
echo Activating virtual environment...
CALL %VENV_DIR%\Scripts\activate.bat

REM Install required packages
IF EXIST requirements.txt (
    echo Installing dependencies from requirements.txt...
    pip install -r requirements.txt
) ELSE (
    echo requirements.txt not found. Skipping dependency installation.
)

echo Environment setup complete.
