# Minion HTTP Server Docker App

This Docker app provides an HTTP server interface for the Minion protocol using DockerModelRunner and OpenAI clients.

## Build and Run

### Option 1: Using Docker Compose (Recommended)

From the repository root directory:

```bash
# Build and run with docker-compose
docker-compose -f docker-compose.minion.yml up --build

# Run in background
docker-compose -f docker-compose.minion.yml up -d --build
```

### Option 2: Manual Docker Build

From the repository root directory:

```bash
# Build the image
docker build -f apps/minions-docker/Dockerfile.minion -t minion-http-server .

# Run in development mode
docker run -d --name minion-server -p 5000:5000 \
  -e OPENAI_API_KEY=YOUR_API_KEY \
  minion-http-server

# Run in production mode
docker run -d --name minion-server -p 5000:5000 \
  -e OPENAI_API_KEY=YOUR_API_KEY \
  -e PRODUCTION=true \
  minion-http-server
```

## Environment Variables

Set these environment variables or pass them via the `/start_protocol` endpoint:

### Required
- `OPENAI_API_KEY` - Your OpenAI API key (required)

### Model Configuration
- `REMOTE_MODEL_NAME` - OpenAI model name (default: "gpt-4o-mini")
- `LOCAL_MODEL_NAME` - Local model name (default: "ai/smollm2")
- `LOCAL_BASE_URL` - Local model base URL (default: "http://model-runner.docker.internal/engines/llama.cpp/v1")
- `REMOTE_BASE_URL` - Remote model base URL (default: "https://api.openai.com/v1")

### Server Configuration
- `MAX_ROUNDS` - Maximum conversation rounds (default: 3)
- `TIMEOUT` - Request timeout in seconds (default: 60)
- `HOST` - Server host (default: "0.0.0.0")
- `PORT` - Server port (default: 5000)
- `LOG_DIR` - Log directory (default: "minion_logs")

### Deployment Mode
- `PRODUCTION` - Run in production mode with Gunicorn (default: false)
- `DEBUG` - Enable debug mode for development (default: false)

## API Endpoints

### Health Check
Check server status and configuration:
```bash
curl http://127.0.0.1:5000/health
```

#### Initialize Protocol
```bash
curl -X POST http://127.0.0.1:5000/start_protocol \
  -H "Content-Type: application/json" \
  -d '{
    "openai_api_key": "YOUR-API-KEY"
  }'
```

#### Run Query
Execute a minion query with context:
```bash
curl -X POST http://127.0.0.1:5000/run \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of this text?",
    "context": ["This is a sample document about machine learning and AI."]
  }'
```

### Configuration Management
Get current configuration:
```bash
curl http://127.0.0.1:5000/config
```

Update configuration:
```bash
curl -X POST http://127.0.0.1:5000/config \
  -H "Content-Type: application/json" \
  -d '{
    "max_rounds": 5,
    "timeout": 120
  }'
```

## Cleanup

### Docker Compose
```bash
# Stop and remove containers
docker-compose -f apps/minions-docker/docker-compose.minion.yml down

# Remove volumes as well
docker-compose -f apps/minions-docker/docker-compose.minion.yml down -v
```

### Manual Docker
```bash
# Stop the container
docker stop minion-server

# Remove the container
docker rm minion-server

# Remove the image
docker rmi minion-http-server
```

## Notes

- The container must be built from the repository root directory to access the `minions` package
- The working directory inside the container is `/app`
- Logs are stored in the `minion_logs` directory inside the container
- The server runs on port 5000 by default

