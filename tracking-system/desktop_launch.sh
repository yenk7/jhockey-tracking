#!/bin/bash

# ArUco Tracking Desktop Launcher
# Double-click to start performance tracking (1080p)

PROJECT_PATH="/Users/jhumechatronics/Desktop/mechatronics"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

echo -e "${PURPLE}🚀 ArUco Desktop Launcher${NC}"
echo -e "${BLUE}========================${NC}"

# Navigate to project
cd "$PROJECT_PATH"

# Launch performance mode
echo -e "${GREEN}Starting performance tracking (1080p)...${NC}"
./track_1080p.sh