#!/bin/bash
# Quick start script for Video Editor API

echo "🎬 Starting Video Editor API..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found!"
    echo "📝 Creating from template..."
    cp .env.example .env
    echo "✅ Created .env - Please edit it with your API keys"
    echo "   Then run this script again."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads logs agent_memory

# Check if API keys are set
if grep -q "your_.*_here" .env; then
    echo "⚠️  WARNING: API keys not configured in .env"
    echo "   Please edit .env with your actual API keys"
fi

# Start server
echo "🚀 Starting development server..."
echo "   Open http://localhost:8000 in your browser"
echo ""
uvicorn api_server:app --reload --port 8000
