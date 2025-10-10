#!/bin/bash

# Test Adaptive ArUco Tracking
# This script tests the enhanced tracking with adaptive detectors and recovery

cd "$(dirname "$0")"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${PURPLE}🧪 Testing Adaptive ArUco Tracking${NC}"
echo -e "${BLUE}==================================${NC}"
echo ""
echo -e "${YELLOW}This version includes:${NC}"
echo -e "${BLUE}🎯 Adaptive detectors (Normal/Fast/Recovery)${NC}"
echo -e "${BLUE}⚡ Fast movement detection${NC}"
echo -e "${BLUE}🔄 Lost marker recovery${NC}"
echo -e "${BLUE}📊 Movement speed monitoring${NC}"
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
echo -e "${GREEN}🟢 Starting adaptive tracking test${NC}"
echo -e "${BLUE}Resolution: 1080p (optimal performance)${NC}"
echo -e "${BLUE}Movement threshold: 0.3cm (very sensitive)${NC}"
echo ""
echo -e "${YELLOW}📋 TEST INSTRUCTIONS:${NC}"
echo -e "${BLUE}1. Start with markers stationary${NC}"
echo -e "${BLUE}2. Move markers slowly${NC}"
echo -e "${BLUE}3. Move markers quickly${NC}"
echo -e "${BLUE}4. Hide markers briefly, then show again${NC}"
echo -e "${BLUE}5. Watch terminal for detector mode changes${NC}"
echo ""

# Run with very sensitive settings for testing
python3 main.py --resolution 1080p --movement-threshold 0.3