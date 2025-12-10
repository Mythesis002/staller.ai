@echo off
REM Quick start script for Video Editor API (Windows)

echo 🎬 Starting Video Editor API...

REM Check if .env exists
if not exist .env (
    echo ⚠️  .env file not found!
    echo 📝 Creating from template...
    copy .env.example .env
    echo ✅ Created .env - Please edit it with your API keys
    echo    Then run this script again.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist .venv (
    echo 📦 Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install/update dependencies
echo 📥 Installing dependencies...
pip install -q -r requirements.txt

REM Create necessary directories
echo 📁 Creating directories...
if not exist uploads mkdir uploads
if not exist logs mkdir logs
if not exist agent_memory mkdir agent_memory

REM Check if API keys are set
findstr /C:"your_" .env >nul
if %errorlevel% equ 0 (
    echo ⚠️  WARNING: API keys not configured in .env
    echo    Please edit .env with your actual API keys
)

REM Start server
echo 🚀 Starting development server...
echo    Open http://localhost:8000 in your browser
echo.
python -m uvicorn api_server:app --reload --port 8000
