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

# Run the container
docker run -d --name minion-server -p 5000:5000 \
  -e OPENAI_API_KEY=YOUR_API_KEY \
  minion-http-server
```

## Environment Variables

Set these environment variables or pass them via the `/start_protocol` endpoint:

- `OPENAI_API_KEY` - Your OpenAI API key (required)
- `REMOTE_MODEL_NAME` - OpenAI model name (default: "gpt-4o-mini")
- `LOCAL_MODEL_NAME` - Local model name (default: "ai/smollm2")
- `LOCAL_BASE_URL` - Local model base URL (default: "http://model-runner.docker.internal/engines/llama.cpp/v1")
- `REMOTE_BASE_URL` - Remote model base URL (default: "http://model-runner.docker.internal/engines/openai/v1")
- `MAX_ROUNDS` - Maximum conversation rounds (default: 3)
- `TIMEOUT` - Request timeout in seconds (default: 60)

## API Endpoints

#### Check Health
```bash
curl http://127.0.0.1:5000/health
```

#### Initialize Protocol
```bash
curl -X POST http://127.0.0.1:5000/start_protocol \
  -H "Content-Type: application/json" \
  -d '{
    "openai_api_key": "YOUR-API-KEY",
    "remote_model_name": "gpt-4o-mini",
    "local_model_name": "ai/smollm2",
    "local_base_url": "http://model-runner.docker.internal/engines/llama.cpp/v1",
    "remote_base_url": "http://model-runner.docker.internal/engines/openai/v1"
  }'
```

#### Run Query
```bash
curl -X POST http://127.0.0.1:5000/run \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of this text?",
    "context": ["This is a sample document about machine learning and AI."]
  }'
```

#### Get/Update Configuration
```bash
# Get current configuration
curl http://127.0.0.1:5000/config

# Update configuration
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
docker-compose -f docker-compose.minion.yml down

# Remove volumes as well
docker-compose -f docker-compose.minion.yml down -v
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
- The working directory inside the container is `/app/apps/minions-docker`
- Logs are stored in the `minion_logs` directory inside the container
- The server runs on port 5000 by default

