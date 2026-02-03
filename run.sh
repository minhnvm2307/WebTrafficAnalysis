#!/bin/bash
echo "🚀 Starting Traffic Analysis System..."
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "📦 Activating virtual environment..."
    source .venv/bin/activate
fi


# Start ML Server in background
echo ""
echo "🤖 Starting ML Server on port 8000..."
uvicorn traffic_monitor.server.ml_server:app --reload --port 8000 &
ML_SERVER_PID=$!

# Wait for ML server to start
echo "⏳ Waiting for ML Server to initialize..."
sleep 3

# Check if ML server is running
if ps -p $ML_SERVER_PID > /dev/null; then
    echo "✅ ML Server running (PID: $ML_SERVER_PID)"
else
    echo "❌ ML Server failed to start"
    exit 1
fi

# Start Streamlit app
echo ""
echo "Starting Streamlit Dashboard on port 8501..."
streamlit run traffic_monitor/app.py &
STREAMLIT_PID=$!

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ System Started Successfully!"
echo ""
echo "📊 Streamlit Dashboard: http://localhost:8501"
echo "🤖 ML Server API:       http://localhost:8000"
echo ""
echo "Process IDs:"
echo "  - ML Server:   $ML_SERVER_PID"
echo "  - Streamlit:   $STREAMLIT_PID"
echo ""
echo "To stop all services:"
echo "  kill $ML_SERVER_PID $STREAMLIT_PID"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Keep script running and handle Ctrl+C
trap "echo ''; echo '🛑 Shutting down...'; kill $ML_SERVER_PID $STREAMLIT_PID 2>/dev/null; exit" INT TERM

# Wait for both processes
wait