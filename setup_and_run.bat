@echo off
REM One-click setup and run for Windows. Needs Python on PATH and internet on the first run.
REM Written without IF blocks on purpose: a bracket inside an echo line breaks a block and closes the window.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 goto :nopython

if exist ".venv\Scripts\python.exe" goto :venv_ok
echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 goto :fail
:venv_ok

if exist ".venv\.installed" goto :installed
echo Installing packages. This takes a few minutes and about 500 MB.
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
echo ok> ".venv\.installed"
:installed

echo Downloading and verifying models, about 35 MB on the first run...
".venv\Scripts\python.exe" download_models.py
if errorlevel 1 goto :fail

echo.
echo Starting the app. Your browser will open. Allow camera access when asked.
echo Keep this window open while you use the app. Close it to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py
echo.
echo The app has stopped.
pause
goto :eof

:nopython
echo Python was not found. Install it from python.org and tick "Add python.exe to PATH", then run this again.
pause
goto :eof

:fail
echo.
echo Setup failed. Scroll up to read the error message above.
pause
goto :eof
