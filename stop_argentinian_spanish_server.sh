#!/bin/bash
# Stop script for Argentinian Spanish XTTS-v2 TTS Server

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping Argentinian Spanish XTTS-v2 Server...${NC}"

if pgrep -f "oai_server.*argentinian_spanish" > /dev/null; then
    pkill -f "oai_server.*argentinian_spanish"
    sleep 2
    
    if pgrep -f "oai_server.*argentinian_spanish" > /dev/null; then
        echo -e "${YELLOW}Server still running, forcing shutdown...${NC}"
        pkill -9 -f "oai_server.*argentinian_spanish"
        sleep 1
    fi
    
    if ! pgrep -f "oai_server.*argentinian_spanish" > /dev/null; then
        echo -e "${GREEN}✓${NC} Server stopped successfully"
    else
        echo -e "${RED}✗${NC} Failed to stop server"
        exit 1
    fi
else
    echo -e "${YELLOW}Server is not running${NC}"
fi
