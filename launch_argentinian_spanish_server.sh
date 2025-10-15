#!/bin/bash
# Launcher for Argentinian Spanish XTTS-v2 TTS Server
# This script starts the OpenAI-compatible TTS server with the Argentinian Spanish model

set -e

# Configuration
MODEL_PATH="/home/op/Auralis/converted_models/argentinian_spanish/core_xttsv2"
GPT_MODEL_PATH="/home/op/Auralis/converted_models/argentinian_spanish/gpt"
HOST="0.0.0.0"
PORT="5000"
MAX_CONCURRENCY="8"
LOG_FILE="/home/op/Auralis/argentinian_spanish_server.log"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Argentinian Spanish XTTS-v2 Server${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Miniconda/Anaconda.${NC}"
    exit 1
fi

# Check if auralis_env exists
if ! conda env list | grep -q "auralis_env"; then
    echo -e "${RED}Error: auralis_env conda environment not found.${NC}"
    echo -e "${YELLOW}Please create the environment first.${NC}"
    exit 1
fi

# Check if model files exist
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model not found at $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -d "$GPT_MODEL_PATH" ]; then
    echo -e "${RED}Error: GPT model not found at $GPT_MODEL_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Model found: $MODEL_PATH"
echo -e "${GREEN}✓${NC} GPT model found: $GPT_MODEL_PATH"
echo ""

# Kill any existing server
echo -e "${YELLOW}Checking for existing server...${NC}"
if pgrep -f "oai_server.*argentinian_spanish" > /dev/null; then
    echo -e "${YELLOW}Stopping existing server...${NC}"
    pkill -f "oai_server.*argentinian_spanish" || true
    sleep 2
fi

echo -e "${GREEN}Starting server...${NC}"
echo -e "  Host: ${BLUE}$HOST${NC}"
echo -e "  Port: ${BLUE}$PORT${NC}"
echo -e "  Concurrency: ${BLUE}$MAX_CONCURRENCY${NC}"
echo -e "  Log file: ${BLUE}$LOG_FILE${NC}"
echo ""

# Activate conda environment and start server
source ~/miniconda3/etc/profile.d/conda.sh
conda activate auralis_env

# Start server in background
nohup python -u -m auralis.entrypoints.oai_server \
    --model "$MODEL_PATH" \
    --gpt_model "$GPT_MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --max_concurrency "$MAX_CONCURRENCY" \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo -e "${GREEN}✓${NC} Server started with PID: ${BLUE}$SERVER_PID${NC}"
echo ""

# Wait for server to start
echo -e "${YELLOW}Waiting for server to initialize...${NC}"
sleep 15

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    # Check if port is listening
    if netstat -tuln 2>/dev/null | grep -q ":$PORT " || ss -tuln 2>/dev/null | grep -q ":$PORT "; then
        echo -e "${GREEN}✓${NC} Server is running and listening on port $PORT"
        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  Server Ready!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo -e "API Endpoint: ${BLUE}http://localhost:$PORT/v1/audio/speech${NC}"
        echo -e "Documentation: ${BLUE}http://localhost:$PORT/docs${NC}"
        echo ""
        echo -e "To view logs: ${YELLOW}tail -f $LOG_FILE${NC}"
        echo -e "To stop server: ${YELLOW}pkill -f 'oai_server.*argentinian_spanish'${NC}"
        echo ""
    else
        echo -e "${RED}✗${NC} Server process running but port not listening"
        echo -e "${YELLOW}Check logs: tail -f $LOG_FILE${NC}"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} Server failed to start"
    echo -e "${YELLOW}Check logs: tail -f $LOG_FILE${NC}"
    exit 1
fi
