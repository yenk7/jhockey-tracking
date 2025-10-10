#!/bin/bash

# Simple Dynamic ArUco Tracking with Recovery
# Back to the working version with just simple recovery added

cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🎯 Simple Dynamic Tracking + Recovery${NC}"
echo -e "${BLUE}====================================${NC}"
echo ""
echo -e "${YELLOW}Back to working dynamic tracking with:${NC}"
echo -e "${BLUE}✅ Dynamic-friendly parameters${NC}"
echo -e "${BLUE}✅ Simple recovery mode${NC}"
echo -e "${BLUE}✅ No complex adaptive logic${NC}"
echo ""

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
fi

# Open browser
echo -e "${YELLOW}🌐 Opening visualization...${NC}"
if command -v open >/dev/null 2>&1; then
    open "file://${PWD}/visualize.html"
fi

echo ""
echo -e "${GREEN}🟢 Starting simple dynamic tracking${NC}"
echo -e "${BLUE}Resolution: 1080p (proven working)${NC}"
echo -e "${BLUE}Movement threshold: 0.5cm${NC}"
echo -e "${BLUE}Recovery: Auto-detects lost markers${NC}"
echo ""

# Run the simplified dynamic tracking
python3 main.py --resolution 1080p --movement-threshold 0.5