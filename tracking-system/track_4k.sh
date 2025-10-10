#!/bin/bash

# ArUco Tracking - 4K Resolution (Maximum Accuracy)
# Best for: Precision tracking, detailed detection, large tracking area

cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🎯 ArUco Tracking - 4K PRECISION MODE${NC}"
echo -e "${BLUE}====================================${NC}"
echo -e "${BLUE}📹 Resolution: 4K (3840x2160)${NC}"
echo -e "${BLUE}🎯 Movement threshold: 1.0cm${NC}"
echo -e "${BLUE}⚡ Frame rate: ~20-25 FPS${NC}"
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
echo -e "${GREEN}🟢 Starting 4K precision tracking...${NC}"
echo -e "${YELLOW}Best for: High accuracy, detailed corner detection${NC}"
echo ""

# Run 4K tracking
python3 main.py --resolution 4k --movement-threshold 1.0