# 🚀 Server Deployment Guide

This guide covers deploying Auralis backend and frontend services on a server.

## Overview

Auralis provides two main components:
- **Backend API Server**: OpenAI-compatible FastAPI server for TTS generation
- **Frontend Gradio UI**: Web interface for interactive voice cloning

## Prerequisites

- Python 3.10+
- Conda environment manager
- NVIDIA GPU with CUDA support (recommended)
- Sufficient GPU memory (minimum 8GB, recommended 16GB+)

## Quick Deployment

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n auralis_env python=3.10 -y
conda activate auralis_env

# Install Auralis
pip install auralis
```

### 2. Launch Backend Server

The backend server provides OpenAI-compatible API endpoints.

```bash
# Default configuration (0.0.0.0:8000)
python -m auralis.entrypoints.oai_server

# Custom configuration
python -m auralis.entrypoints.oai_server \
  --host 0.0.0.0 \
  --port 8000 \
  --model AstraMindAI/xttsv2 \
  --gpt_model AstraMindAI/xtts2-gpt \
  --max_concurrency 8 \
  --vllm_logging_level warn
```

**Configuration Options:**
- `--host`: Server host (default: `0.0.0.0`)
- `--port`: Server port (default: `8000`)
- `--model`: Base TTS model path
- `--gpt_model`: GPT model path
- `--max_concurrency`: Maximum concurrent requests (affects GPU memory)
- `--vllm_logging_level`: Logging level (`info`, `warn`, `err`)

### 3. Launch Frontend UI

The Gradio interface provides an interactive web UI.

```bash
cd examples
python gradio_example.py
```

**Default Configuration:**
- Host: `0.0.0.0` (accessible from any network interface)
- Port: `7860` (Gradio default)
- Concurrency: `4` (optimized for GPU memory sharing with backend)

## Production Deployment

### Using systemd Services

#### Backend Service

Create `/etc/systemd/system/auralis-backend.service`:

```ini
[Unit]
Description=Auralis TTS Backend Server
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/Auralis
Environment="PATH=/home/your-username/miniconda3/envs/auralis_env/bin"
ExecStart=/home/your-username/miniconda3/envs/auralis_env/bin/python -m auralis.entrypoints.oai_server --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Frontend Service

Create `/etc/systemd/system/auralis-frontend.service`:

```ini
[Unit]
Description=Auralis Gradio Frontend
After=network.target auralis-backend.service

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/Auralis/examples
Environment="PATH=/home/your-username/miniconda3/envs/auralis_env/bin"
ExecStart=/home/your-username/miniconda3/envs/auralis_env/bin/python gradio_example.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### Enable and Start Services

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable services to start on boot
sudo systemctl enable auralis-backend
sudo systemctl enable auralis-frontend

# Start services
sudo systemctl start auralis-backend
sudo systemctl start auralis-frontend

# Check status
sudo systemctl status auralis-backend
sudo systemctl status auralis-frontend

# View logs
sudo journalctl -u auralis-backend -f
sudo journalctl -u auralis-frontend -f
```

### Using Docker

#### Dockerfile

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Auralis
RUN pip install auralis

# Expose ports
EXPOSE 8000 7860

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Default command
CMD ["python3", "-m", "auralis.entrypoints.oai_server", "--host", "0.0.0.0", "--port", "8000"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  auralis-backend:
    build: .
    ports:
      - "8000:8000"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    command: python3 -m auralis.entrypoints.oai_server --host 0.0.0.0 --port 8000

  auralis-frontend:
    build: .
    ports:
      - "7860:7860"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    depends_on:
      - auralis-backend
    working_dir: /app/examples
    command: python3 gradio_example.py
```

Run with:
```bash
docker-compose up -d
```

## GPU Memory Management

When running both services on the same GPU:

### Backend Configuration
- Default `max_concurrency=8` uses ~18% GPU memory
- Adjust based on available VRAM:
  - 8GB VRAM: `max_concurrency=2-4`
  - 16GB VRAM: `max_concurrency=6-8`
  - 24GB VRAM: `max_concurrency=10-16`

### Frontend Configuration
- Default `scheduler_max_concurrency=4` uses ~26% GPU memory
- Modify in `gradio_example.py`:
  ```python
  tts = TTS(scheduler_max_concurrency=4)  # Adjust this value
  ```

### Combined Usage
Total GPU memory = Backend + Frontend + Model weights (~2.5GB base)

Example configurations:
- **24GB GPU**: Backend (8) + Frontend (4) = ~44% usage
- **16GB GPU**: Backend (4) + Frontend (2) = ~35% usage
- **8GB GPU**: Backend (2) + Frontend (1) = ~30% usage

## Nginx Reverse Proxy

```nginx
# Backend API
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Frontend UI
server {
    listen 80;
    server_name tts.yourdomain.com;

    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Monitoring

### Health Checks

Backend:
```bash
curl http://localhost:8000/docs
```

Frontend:
```bash
curl http://localhost:7860
```

### Log Locations

When running as systemd services:
```bash
# Backend logs
sudo journalctl -u auralis-backend -n 100

# Frontend logs
sudo journalctl -u auralis-frontend -n 100
```

When running manually:
```bash
# Redirect to files
python -m auralis.entrypoints.oai_server > backend.log 2>&1 &
python gradio_example.py > frontend.log 2>&1 &
```

## Troubleshooting

### Port Already in Use
```bash
# Find process using port
lsof -i :8000
lsof -i :7860

# Kill process
kill -9 <PID>
```

### GPU Memory Issues
```bash
# Check GPU usage
nvidia-smi

# Clear GPU memory
pkill -9 python
```

### Service Won't Start
```bash
# Check logs
sudo journalctl -u auralis-backend -n 50
sudo journalctl -u auralis-frontend -n 50

# Verify conda environment
conda activate auralis_env
which python
python --version
```

## Security Considerations

1. **Firewall Configuration**
   ```bash
   # Allow only specific IPs
   sudo ufw allow from YOUR_IP to any port 8000
   sudo ufw allow from YOUR_IP to any port 7860
   ```

2. **API Authentication**
   - Consider adding API key authentication
   - Use HTTPS in production
   - Implement rate limiting

3. **Resource Limits**
   - Set `max_concurrency` to prevent resource exhaustion
   - Monitor GPU memory usage
   - Implement request timeouts

## Performance Optimization

1. **Model Caching**: Models are cached after first load
2. **Concurrent Requests**: Adjust `max_concurrency` based on workload
3. **Streaming**: Use streaming for long texts to reduce latency
4. **GPU Selection**: Use `CUDA_VISIBLE_DEVICES` to specify GPU

```bash
# Use specific GPU
CUDA_VISIBLE_DEVICES=0 python -m auralis.entrypoints.oai_server
```

## Support

- [GitHub Issues](https://github.com/astramind-ai/Auralis/issues)
- [Discord Community](https://discord.gg/BEMVTmcPEs)
- [Documentation](https://github.com/astramind-ai/Auralis/tree/main/docs)
