@echo off
REM ---------------------------------------------------------------------------
REM One-command local setup for Windows. Double-click this file, or run:
REM
REM     setup.bat
REM
REM What it does, and it is safe to run again at any time:
REM   1. creates a Python virtual environment in .venv\
REM   2. installs the pinned requirements
REM   3. asks for your Groq API key once and saves it in .streamlit\secrets.toml
REM      (that file is gitignored, so the key can never be committed)
REM   4. starts the chatbot at http://localhost:8501
REM
REM The first start builds the search index from data\raw\resume.txt, which
REM takes about a minute. Later starts are instant until you edit the resume.
REM ---------------------------------------------------------------------------
setlocal
cd /d "%~dp0"

set "SECRETS=.streamlit\secrets.toml"
set "RESUME=data\raw\resume.txt"

REM ---- 1. Python -------------------------------------------------------------
REM runtime.txt pins 3.12. Prefer the "py" launcher's 3.12, then any python.
set "PY="
py -3.12 --version >nul 2>&1 && set "PY=py -3.12"
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo Python was not found. Install Python 3.12 from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" during install. Then run this again.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ==^> Creating virtual environment ^(.venv^)
    %PY% -m venv .venv || goto :fail
)
set "VPY=.venv\Scripts\python.exe"

REM ---- 2. Requirements -------------------------------------------------------
echo.
echo ==^> Installing requirements ^(first time takes a few minutes^)
"%VPY%" -m pip install --upgrade pip --quiet || goto :fail
"%VPY%" -m pip install -r requirements.txt --quiet || goto :fail

REM ---- 3. Resume -------------------------------------------------------------
if not exist "%RESUME%" (
    echo [!] No %RESUME% found. Copying the template; edit it with your own details.
    copy /y "data\raw\resume.example.txt" "%RESUME%" >nul
) else (
    git diff --quiet HEAD -- "%RESUME%" >nul 2>&1
    if not errorlevel 1 (
        echo [!] %RESUME% is still the original author's resume.
        echo [!] Replace it with yours ^(start from data\raw\resume.example.txt^), then re-run.
    )
)

REM ---- 4. API key ------------------------------------------------------------
if exist "%SECRETS%" (
    echo Using existing %SECRETS%
    goto :run
)
echo.
echo ==^> Groq API key
echo Create a free key at https://console.groq.com/keys ^(it starts with gsk_^).
:askkey
set "KEY="
set /p "KEY=Paste your key and press Enter: "
if not defined KEY goto :askkey
if not exist ".streamlit" mkdir ".streamlit"
> "%SECRETS%" echo # Created by setup.bat. Gitignored - never commit this file.
>> "%SECRETS%" echo GROQ_API_KEY = "%KEY%"
>> "%SECRETS%" echo EXPERIMENT = "my_profile"
echo Saved to %SECRETS%
cls
echo Key saved to %SECRETS%

REM ---- 5. Run ----------------------------------------------------------------
:run
echo.
echo ==^> Starting the chatbot at http://localhost:8501  ^(Ctrl+C to stop^)
"%VPY%" -m streamlit run ui\app.py
goto :eof

:fail
echo.
echo Setup failed - see the error above. If it mentions your Python version,
echo install Python 3.12 from https://www.python.org/downloads/ and run this again.
pause
exit /b 1
