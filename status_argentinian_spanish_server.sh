#!/bin/bash
# Status checker for Argentinian Spanish XTTS-v2 TTS Server

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

PORT="5000"
LOG_FILE="/home/op/Auralis/argentinian_spanish_server.log"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Server Status Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if process is running
if pgrep -f "oai_server.*argentinian_spanish" > /dev/null; then
    PID=$(pgrep -f "oai_server.*argentinian_spanish")
    echo -e "Status: ${GREEN}RUNNING${NC}"
    echo -e "PID: ${BLUE}$PID${NC}"
    
    # Check port
    if netstat -tuln 2>/dev/null | grep -q ":$PORT " || ss -tuln 2>/dev/null | grep -q ":$PORT "; then
        echo -e "Port $PORT: ${GREEN}LISTENING${NC}"
    else
        echo -e "Port $PORT: ${RED}NOT LISTENING${NC}"
    fi
    
    # Memory usage
    if [ -n "$PID" ]; then
        MEM=$(ps -p $PID -o rss= | awk '{printf "%.2f GB", $1/1024/1024}')
        echo -e "Memory: ${BLUE}$MEM${NC}"
    fi
    
    # Uptime
    if [ -n "$PID" ]; then
        UPTIME=$(ps -p $PID -o etime= | tr -d ' ')
        echo -e "Uptime: ${BLUE}$UPTIME${NC}"
    fi
    
    echo ""
    echo -e "API Endpoint: ${BLUE}http://localhost:$PORT/v1/audio/speech${NC}"
    echo -e "Documentation: ${BLUE}http://localhost:$PORT/docs${NC}"
    echo ""
    echo -e "Recent logs (last 10 lines):"
    echo -e "${YELLOW}---${NC}"
    tail -10 "$LOG_FILE" 2>/dev/null || echo "No logs available"
    echo -e "${YELLOW}---${NC}"
    
else
    echo -e "Status: ${RED}NOT RUNNING${NC}"
    echo ""
    echo -e "To start server: ${GREEN}./launch_argentinian_spanish_server.sh${NC}"
fi

echo ""
