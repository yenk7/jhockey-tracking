#!/bin/bash

set -e

echo "🚀 Testing Improved Recovery System"
echo "=================================="
echo ""
echo "👉 Instructions:"
echo "  1. Show your ArUco markers to the camera"
echo "  2. Wait a few seconds so the system detects them"
echo "  3. Hide/remove the markers for at least 2 seconds"
echo "  4. Show the markers again"
echo "  5. Watch the terminal for recovery messages"
echo ""
echo "🔍 Starting tracking with improved recovery..."

# Activate virtual environment if available
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate || true
fi

python3 main.py --resolution 1080p --movement-threshold 0.5

echo "Test completed!"