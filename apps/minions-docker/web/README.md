# Minions Docker Web Interface

## 🚀 Quick Start

### **With Docker Compose (Recommended)**
```bash
# From apps/minions-docker/ directory
OPENAI_API_KEY=sk-your-key docker compose up

# Access frontend at: http://localhost:8080
```

### **Local Development**
```bash
# Install dependencies
cd apps/minions-docker/web
npm install

# Start development server
npm run dev

# Access at: http://localhost:8080
```

## 🔧 Features

### **Simplified Interface**
- **Auto-Configuration**: Automatically detects backend URL
- **No Manual Setup**: Works out of the box with Docker Compose
- **Smart Detection**: Uses `127.0.0.1:5000` to avoid macOS AirPlay conflicts

### **Core Functionality**
- **Task Input**: Enter queries and questions
- **Context Support**: Provide supporting documents
- **Execution Control**: Configure max rounds and timeout
- **Real-time Logs**: View execution progress
- **Metrics Display**: See token usage and timing information

### **Backend Integration**
- **Health Checks**: Automatic backend status monitoring
- **Error Handling**: Clear error messages and troubleshooting
- **CORS Support**: Properly configured for cross-origin requests

## 🌐 Backend Connection

The frontend automatically detects the backend URL based on the environment:

- **Docker Compose**: `http://127.0.0.1:5000` (avoids macOS AirPlay conflict)
- **Local Development**: `http://127.0.0.1:5001` (alternative port)
- **Container Mode**: Uses Docker service names for internal communication

## 📋 Configuration

### **Environment Variables**
- `BACKEND_URL`: Override backend URL (set by Docker Compose)
- `NODE_ENV`: Development/production mode
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8080)

### **Execution Settings**
- **Max Rounds**: 3, 5, 7, or 10 protocol rounds
- **Timeout**: 60, 120, 180, or 300 seconds

## 🔍 Troubleshooting

### **Common Issues**

#### **Backend Connection Failed**
```bash
# Check if backend is running
curl http://127.0.0.1:5000/health

# Start backend if needed
cd apps/minions-docker
OPENAI_API_KEY=sk-your-key docker compose up minion-server
```

#### **macOS Port 5000 Conflict**
The frontend automatically uses `127.0.0.1:5000` instead of `localhost:5000` to avoid AirPlay conflicts.

#### **CORS Errors**
The backend is configured to accept requests from common development ports. If you're using a custom port, update the CORS configuration in `minion_http_server.py`.

## 🎯 Usage Examples

### **Basic Task**
1. Open http://localhost:8080
2. Enter task: "Summarize the key points"
3. Add context: "Your document content here"
4. Click "Start Minion Protocol"

### **Advanced Configuration**
1. Set Max Rounds to 5 for complex tasks
2. Increase timeout to 180s for long-running tasks
3. Monitor execution logs and metrics

The web interface provides a complete, user-friendly way to interact with the Minions protocol without needing to use curl or write custom scripts.
