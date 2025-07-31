// main.js - Minions Docker Web Interface
class MinionsDockerClient {
  constructor() {
    this.backendUrl = this.detectBackendUrl();
    this.isInitialized = false;
    this.initializeElements();
    this.updateBackendUrlDisplay();
    this.attachEventListeners();
    this.checkBackendStatus();
  }

  detectBackendUrl() {
    // Check if we're running in Docker (via environment variable or hostname detection)
    const hostname = window.location.hostname;
    
    // If running in Docker Compose, use the service name
    if (hostname !== 'localhost' && hostname !== '127.0.0.1') {
      // Running in Docker - use internal service URL
      return `http://${hostname}:5000`;
    }
    
    // Check for different local development ports
    const currentPort = window.location.port;
    if (currentPort === '8080') {
      // Running via Docker Compose - use 127.0.0.1 to force IPv4 and avoid AirPlay conflict
      return 'http://127.0.0.1:5000';
    }
    
    // Default to port 5001 for local development (avoiding AirTunes conflict)
    return 'http://127.0.0.1:5001';
  }

  initializeElements() {
    // Get DOM elements
    this.elements = {
      statusCard: document.getElementById('status-card'),
      maxRounds: document.getElementById('max_rounds'),
      timeout: document.getElementById('timeout'),
      task: document.getElementById('task'),
      context: document.getElementById('context'),
      checkStatusBtn: document.getElementById('check_status'),
      startBtn: document.getElementById('start'),
      clearLogBtn: document.getElementById('clear_log'),
      log: document.getElementById('log'),
      metricsContainer: document.getElementById('metrics-container'),
      metrics: document.getElementById('metrics')
    };
  }

  attachEventListeners() {
    this.elements.checkStatusBtn.onclick = () => this.checkBackendStatus();
    this.elements.startBtn.onclick = () => this.startMinion();
    this.elements.clearLogBtn.onclick = () => this.clearLog();
    
    // Update start button state when task changes
    this.elements.task.oninput = () => this.updateStartButtonState();
  }

  updateStatus(status, message, emoji = '🔄') {
    const statusCard = this.elements.statusCard;
    statusCard.className = `status-card status-${status}`;
    statusCard.innerHTML = `<span>${emoji}</span><span>${message}</span>`;
  }

  updateBackendUrlDisplay() {
    const backendUrlElement = document.getElementById('backend-url-display');
    if (backendUrlElement) {
      backendUrlElement.textContent = this.backendUrl;
    }
  }

  updateStartButtonState() {
    const task = this.elements.task.value.trim();
    
    this.elements.startBtn.disabled = !task;
  }

  logMessage(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const prefix = {
      'info': '📝',
      'success': '✅',
      'error': '❌',
      'warning': '⚠️'
    }[type] || '📝';
    
    const logElement = this.elements.log;
    logElement.textContent += `[${timestamp}] ${prefix} ${message}\n`;
    logElement.scrollTop = logElement.scrollHeight;
  }

  clearLog() {
    this.elements.log.textContent = 'Log cleared.\n';
    this.elements.metricsContainer.classList.add('hidden');
  }

  async makeRequest(endpoint, options = {}) {
    const url = `${this.backendUrl}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers
        },
        ...options
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.message || data.error || `HTTP ${response.status}`);
      }
      
      return data;
    } catch (error) {
      if (error.name === 'TypeError' && error.message.includes('fetch')) {
        throw new Error(`Cannot connect to backend at ${url}. Make sure the server is running.`);
      }
      throw error;
    }
  }

  async checkBackendStatus() {
    this.updateStatus('warning', 'Checking backend status...', '🔄');
    
    try {
      const data = await this.makeRequest('/health');
      
      if (data.status === 'healthy') {
        const modelInfo = data.config.remote_model_name ? ` (${data.config.remote_model_name})` : '';
        this.updateStatus('healthy', `Backend is healthy${modelInfo}`, '✅');
        this.isInitialized = data.config.minion_initialized || true; // Assume initialized if backend is healthy
        
        // Update max rounds from backend configuration if available
        if (data.config.max_rounds) {
          this.elements.maxRounds.value = data.config.max_rounds.toString();
        }
        
        this.logMessage(`Backend health check successful. Ready to process tasks.`, 'success');
        this.logMessage(`Configuration: Remote=${data.config.remote_model_name || 'N/A'}, Local=${data.config.local_model_name || 'N/A'}`, 'info');
      } else {
        this.updateStatus('warning', 'Backend responded but status unknown', '⚠️');
        this.logMessage('Backend health check returned unknown status', 'warning');
      }
    } catch (error) {
      this.updateStatus('error', `Backend connection failed: ${error.message}`, '❌');
      this.logMessage(`Backend health check failed: ${error.message}`, 'error');
      this.isInitialized = false;
    }
    
    this.updateStartButtonState();
  }

  async startMinion() {
    const task = this.elements.task.value.trim();
    const context = this.elements.context.value.trim();

    if (!task) {
      this.logMessage('Task is required', 'error');
      return;
    }

    // Disable start button and show loading state
    this.elements.startBtn.disabled = true;
    this.elements.startBtn.innerHTML = '<span class="spinner"></span>Running...';

    const startTime = Date.now();

    try {
      this.logMessage('Starting minion protocol...', 'info');
      this.logMessage(`Task: ${task}`, 'info');
      if (context) {
        this.logMessage(`Context length: ${context.length} characters`, 'info');
      }

      const requestData = {
        query: task,
        context: context ? [context] : [],
        max_rounds: parseInt(this.elements.maxRounds.value),
        logging_id: `web_${Date.now()}`
      };

      const data = await this.makeRequest('/run', {
        method: 'POST',
        body: JSON.stringify(requestData)
      });

      const endTime = Date.now();
      const executionTime = (endTime - startTime) / 1000;

      this.logMessage('Minion protocol completed successfully!', 'success');
      this.logMessage('='.repeat(50), 'info');
      this.logMessage('FINAL ANSWER:', 'success');
      this.logMessage(data.final_answer, 'info');
      this.logMessage('='.repeat(50), 'info');

      // Display metrics
      this.displayMetrics({
        executionTime,
        remoteUsage: data.usage.remote,
        localUsage: data.usage.local,
        timing: data.timing,
        logFile: data.log_file
      });

    } catch (error) {
      this.logMessage(`Minion execution failed: ${error.message}`, 'error');
    } finally {
      this.elements.startBtn.disabled = false;
      this.elements.startBtn.innerHTML = 'Start Minion Protocol';
      this.updateStartButtonState();
    }
  }

  displayMetrics(metrics) {
    const metricsContainer = this.elements.metricsContainer;
    const metricsElement = this.elements.metrics;
    
    // Clear previous metrics
    metricsElement.innerHTML = '';
    
    // Execution time
    this.addMetric('Execution Time', `${metrics.executionTime.toFixed(2)}s`, metricsElement);
    
    // Remote usage
    if (metrics.remoteUsage) {
      this.addMetric('Remote Tokens', metrics.remoteUsage.total_tokens || 'N/A', metricsElement);
      this.addMetric('Remote Prompt Tokens', metrics.remoteUsage.prompt_tokens || 'N/A', metricsElement);
      this.addMetric('Remote Completion Tokens', metrics.remoteUsage.completion_tokens || 'N/A', metricsElement);
    }
    
    // Local usage
    if (metrics.localUsage) {
      this.addMetric('Local Tokens', metrics.localUsage.total_tokens || 'N/A', metricsElement);
      this.addMetric('Local Prompt Tokens', metrics.localUsage.prompt_tokens || 'N/A', metricsElement);
      this.addMetric('Local Completion Tokens', metrics.localUsage.completion_tokens || 'N/A', metricsElement);
    }
    
    // Timing information
    if (metrics.timing) {
      Object.entries(metrics.timing).forEach(([key, value]) => {
        if (typeof value === 'number') {
          this.addMetric(key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()), 
                        `${value.toFixed(2)}s`, metricsElement);
        }
      });
    }
    
    // Log file
    if (metrics.logFile) {
      this.addMetric('Log File', metrics.logFile, metricsElement);
    }
    
    // Show metrics container
    metricsContainer.classList.remove('hidden');
    
    this.logMessage('Execution metrics displayed below', 'info');
  }

  addMetric(label, value, container) {
    const metricDiv = document.createElement('div');
    metricDiv.className = 'metric-item';
    metricDiv.innerHTML = `
      <div class="metric-value">${value}</div>
      <div class="metric-label">${label}</div>
    `;
    container.appendChild(metricDiv);
  }
}

// Initialize the application when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  new MinionsDockerClient();
});

// Also initialize if the script is loaded after DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new MinionsDockerClient();
  });
} else {
  new MinionsDockerClient();
}
