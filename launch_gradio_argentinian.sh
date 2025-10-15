#!/bin/bash
# Launcher for Argentinian Spanish XTTS-v2 Gradio Interface

set -e

# Configuration
GRADIO_PORT="7861"
LOG_FILE="/home/op/Auralis/gradio_argentinian.log"
SCRIPT_PATH="/home/op/Auralis/gradio_argentinian_spanish_v2.py"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  🇦🇷 Argentinian Spanish Gradio UI${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found.${NC}"
    exit 1
fi

# Check if auralis_env exists
if ! conda env list | grep -q "auralis_env"; then
    echo -e "${RED}Error: auralis_env conda environment not found.${NC}"
    exit 1
fi

# Kill any existing Gradio instance
echo -e "${YELLOW}Checking for existing Gradio instance...${NC}"
if pgrep -f "gradio_argentinian_spanish" > /dev/null; then
    echo -e "${YELLOW}Stopping existing Gradio...${NC}"
    pkill -f "gradio_argentinian_spanish" || true
    sleep 2
fi

echo -e "${GREEN}Starting Gradio interface...${NC}"
echo -e "  Port: ${BLUE}$GRADIO_PORT${NC}"
echo -e "  Log file: ${BLUE}$LOG_FILE${NC}"
echo ""

# Activate conda and start Gradio
source ~/miniconda3/etc/profile.d/conda.sh
conda activate auralis_env

nohup python "$SCRIPT_PATH" > "$LOG_FILE" 2>&1 &
GRADIO_PID=$!

echo -e "${GREEN}✓${NC} Gradio started with PID: ${BLUE}$GRADIO_PID${NC}"
echo ""

# Wait for Gradio to start
echo -e "${YELLOW}Waiting for Gradio to initialize...${NC}"
sleep 20

# Check if Gradio is running
if ps -p $GRADIO_PID > /dev/null; then
    if netstat -tuln 2>/dev/null | grep -q ":$GRADIO_PORT " || ss -tuln 2>/dev/null | grep -q ":$GRADIO_PORT "; then
        echo -e "${GREEN}✓${NC} Gradio is running and accessible!"
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  🎉 Gradio UI Ready!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo -e "🌐 Open in browser: ${BLUE}http://localhost:$GRADIO_PORT${NC}"
        echo ""
        echo -e "📝 Features:"
        echo -e "  • Text to Speech with voice cloning"
        echo -e "  • Record your voice and clone it"
        echo -e "  • Argentinian Spanish examples"
        echo -e "  • Advanced parameter controls"
        echo ""
        echo -e "To view logs: ${YELLOW}tail -f $LOG_FILE${NC}"
        echo -e "To stop Gradio: ${YELLOW}pkill -f 'gradio_argentinian_spanish.py'${NC}"
        echo ""
    else
        echo -e "${RED}✗${NC} Gradio process running but port not listening"
        echo -e "${YELLOW}Check logs: tail -f $LOG_FILE${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} Gradio failed to start"
    echo -e "${YELLOW}Check logs: tail -f $LOG_FILE${NC}"
    exit 1
fi
