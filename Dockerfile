FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app
COPY . .

# Install Python packages
RUN pip install --upgrade pip \
    && pip install -e . \
    && pip install -e .[embeddings] \
    && pip install requests aiohttp

# Install Docker (needed for Docker Model Runner)
RUN curl -fsSL https://get.docker.com -o get-docker.sh \
    && sh get-docker.sh \
    && rm get-docker.sh

# Install Ollama (needed for local model serving)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy the stdin/stdout interface script
COPY minion_stdin_interface.py /app/minion_stdin_interface.py

# Make the script executable
RUN chmod +x /app/minion_stdin_interface.py

# Create log directories
RUN mkdir -p /app/minion_logs

# Set environment variables for local model running
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV OLLAMA_HOST=0.0.0.0:11434

# Expose ports for Docker Model Runner and Ollama
EXPOSE 12434 11434

# Create Ollama startup script
RUN echo '#!/bin/bash\n\
echo "Starting Ollama service..."\n\
ollama serve &\n\
echo "Waiting for Ollama to be ready..."\n\
sleep 5\n\
echo "Checking Ollama status..."\n\
curl -s http://localhost:11434/api/version || echo "Ollama not ready yet"\n\
echo "Ollama service is ready and running in background"\n\
wait\n\
' > /app/start_ollama.sh && chmod +x /app/start_ollama.sh

# Create minion interface script
RUN echo '#!/bin/bash\n\
echo "Starting minion interface..."\n\
python /app/minion_stdin_interface.py\n\
' > /app/start_minion.sh && chmod +x /app/start_minion.sh

# Create main startup script that starts services and keeps running
RUN echo '#!/bin/bash\n\
echo "Initializing services..."\n\
/app/start_ollama.sh &\n\
echo "Waiting for Ollama to be ready..."\n\
sleep 8\n\
echo "Ollama started in background. You can now run /app/start_minion.sh to start the minion interface"\n\
tail -f /dev/null\n\
' > /app/start.sh && chmod +x /app/start.sh

# Create a separate script for piped input mode
RUN echo '#!/bin/bash\n\
echo "Initializing services..."\n\
/app/start_ollama.sh &\n\
echo "Waiting for Ollama to be ready..."\n\
sleep 8\n\
echo "Processing piped input..."\n\
python /app/minion_stdin_interface.py\n\
' > /app/start_piped.sh && chmod +x /app/start_piped.sh

# Default command runs the startup script
CMD ["/app/start.sh"]