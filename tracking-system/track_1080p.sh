#!/bin/bash

# ArUco Tracking - 1080p Resolution (Best Performance)
# Best for: High frame rate, real-time tracking, smooth movement

cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}⚡ ArUco Tracking - 1080p PERFORMANCE MODE${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}📹 Resolution: 1080p (1920x1080)${NC}"
echo -e "${BLUE}🎯 Movement threshold: 0.5cm${NC}"
echo -e "${BLUE}⚡ Frame rate: ~30 FPS${NC}"
echo -e "${GREEN}🔧 Dynamic parameters: Auto-applied${NC}"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
fi

# Install packages if needed
python3 -c "import cv2, numpy, websockets" 2>/dev/null || {
    echo -e "${YELLOW}📦 Installing required packages...${NC}"
    pip install opencv-python numpy websockets uvloop
}

# Open visualization
echo -e "${YELLOW}🌐 Opening visualization...${NC}"
if command -v open >/dev/null 2>&1; then
    open "file://${PWD}/visualize.html"
fi

echo ""
echo -e "${GREEN}🟢 Starting 1080p performance tracking...${NC}"
echo -e "${YELLOW}Best for: Smooth movement, real-time response${NC}"
echo ""

# Run 1080p tracking
python3 main.py --resolution 1080p --movement-threshold 0.5