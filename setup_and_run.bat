@echo off
REM One-click setup + run for Windows. Needs Python 3.10+ on PATH and an internet connection (first run only).
cd /d "%~dp0"

if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv || goto :fail
)
if not exist .venv\.installed (
    echo Installing packages (a few minutes, ~500 MB)...
    .venv\Scripts\python -m pip install --upgrade pip || goto :fail
    .venv\Scripts\python -m pip install -r requirements.txt || goto :fail
    echo ok> .venv\.installed
)
echo Downloading / verifying models (~33 MB)...
.venv\Scripts\python download_models.py || goto :fail

echo.
echo Starting the app. Your browser will open; allow camera access when asked.
.venv\Scripts\streamlit run app.py
goto :eof

:fail
echo.
echo Setup failed. Scroll up for the error message.
pause
