// main.js - Advanced Minions Docker Web Interface
class MinionsAdvancedClient {
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
    
    // Default to port 5000 for local development
    return 'http://127.0.0.1:5000';
  }

  initializeElements() {
    // Get DOM elements
    this.elements = {
      statusCard: document.getElementById('status-card'),
      form: document.getElementById('minions-form'),
      
      // Task configuration
      task: document.getElementById('task'),
      docMetadata: document.getElementById('doc_metadata'),
      context: document.getElementById('context'),
      
      // Basic configuration
      maxRounds: document.getElementById('max_rounds'),
      loggingId: document.getElementById('logging_id'),
      
      // Advanced configuration
      advancedToggle: document.getElementById('advanced-toggle'),
      advancedArrow: document.getElementById('advanced-arrow'),
      advancedContent: document.getElementById('advanced-content'),
      maxJobsPerRound: document.getElementById('max_jobs_per_round'),
      numTasksPerRound: document.getElementById('num_tasks_per_round'),
      numSamplesPerTask: document.getElementById('num_samples_per_task'),
      useRetrieval: document.getElementById('use_retrieval'),
      chunkFn: document.getElementById('chunk_fn'),
      retrievalModel: document.getElementById('retrieval_model'),
      
      // Buttons
      checkStatusBtn: document.getElementById('check_status'),
      startBtn: document.getElementById('start'),
      clearLogBtn: document.getElementById('clear_log'),
      
      // Output
      log: document.getElementById('log'),
      metricsContainer: document.getElementById('metrics-container'),
      metrics: document.getElementById('metrics')
    };
  }

  attachEventListeners() {
    // Button event listeners
    this.elements.checkStatusBtn.onclick = () => this.checkBackendStatus();
    this.elements.startBtn.onclick = (e) => {
      e.preventDefault();
      this.startMinions();
    };
    this.elements.clearLogBtn.onclick = () => this.clearLog();
    
    // Form submission
    this.elements.form.onsubmit = (e) => {
      e.preventDefault();
      this.startMinions();
    };
    
    // Advanced options toggle
    this.elements.advancedToggle.onclick = () => this.toggleAdvancedOptions();
    
    // Update start button state when task changes
    this.elements.task.oninput = () => this.updateStartButtonState();
    
    // Auto-generate logging ID when task changes
    this.elements.task.onblur = () => this.autoGenerateLoggingId();
  }

  toggleAdvancedOptions() {
    const content = this.elements.advancedContent;
    const arrow = this.elements.advancedArrow;
    
    if (content.classList.contains('show')) {
      content.classList.remove('show');
      arrow.textContent = '▶';
    } else {
      content.classList.add('show');
      arrow.textContent = '▼';
    }
  }

  autoGenerateLoggingId() {
    if (!this.elements.loggingId.value.trim() && this.elements.task.value.trim()) {
      const taskWords = this.elements.task.value.trim().split(' ').slice(0, 3).join('_');
      const timestamp = Date.now();
      this.elements.loggingId.value = `${taskWords}_${timestamp}`.toLowerCase().replace(/[^a-z0-9_]/g, '');
    }
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
      'warning': '⚠️',
      'debug': '🔧'
    }[type] || '📝';
    
    const logElement = this.elements.log;
    logElement.textContent += `[${timestamp}] ${prefix} ${message}\n`;
    logElement.scrollTop = logElement.scrollHeight;
  }

  clearLog() {
    this.elements.log.textContent = 'Log cleared.\nWelcome to Advanced Minions Protocol Interface\n';
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
        const modelInfo = data.config?.remote_model_name ? ` (${data.config.remote_model_name})` : '';
        this.updateStatus('healthy', `Backend is healthy${modelInfo}`, '✅');
        this.isInitialized = data.config?.minions_initialized || true;
        
        // Update configuration from backend if available
        if (data.config) {
          if (data.config.max_rounds) {
            this.elements.maxRounds.value = data.config.max_rounds.toString();
          }
          if (data.config.max_jobs_per_round) {
            this.elements.maxJobsPerRound.value = data.config.max_jobs_per_round.toString();
          }
          if (data.config.num_tasks_per_round) {
            this.elements.numTasksPerRound.value = data.config.num_tasks_per_round.toString();
          }
          if (data.config.num_samples_per_task) {
            this.elements.numSamplesPerTask.value = data.config.num_samples_per_task.toString();
          }
          if (data.config.use_retrieval) {
            this.elements.useRetrieval.value = data.config.use_retrieval;
          }
          if (data.config.chunking_function) {
            this.elements.chunkFn.value = data.config.chunking_function;
          }
          if (data.config.retrieval_model) {
            this.elements.retrievalModel.value = data.config.retrieval_model;
          }
        }
        
        this.logMessage(`Backend health check successful. Advanced Minions Protocol ready.`, 'success');
        this.logMessage(`Configuration: Remote=${data.config?.remote_model_name || 'N/A'}, Local=${data.config?.local_model_name || 'N/A'}`, 'info');
        
        // Log available features
        if (data.features) {
          const features = [];
          if (data.features.retrieval_available) features.push('Retrieval');
          if (data.features.bm25_retrieval_available) features.push('BM25');
          if (data.features.openai_available) features.push('OpenAI');
          if (data.features.docker_available) features.push('Docker');
          
          if (features.length > 0) {
            this.logMessage(`Available features: ${features.join(', ')}`, 'info');
          }
        }
        
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

  collectFormData() {
    const task = this.elements.task.value.trim();
    const docMetadata = this.elements.docMetadata.value.trim() || 'Document';
    const context = this.elements.context.value.trim();
    
    // Basic configuration
    const maxRounds = parseInt(this.elements.maxRounds.value) || 3;
    const loggingId = this.elements.loggingId.value.trim() || null;
    
    // Advanced configuration
    const maxJobsPerRound = parseInt(this.elements.maxJobsPerRound.value) || 2048;
    const numTasksPerRound = parseInt(this.elements.numTasksPerRound.value) || 3;
    const numSamplesPerTask = parseInt(this.elements.numSamplesPerTask.value) || 1;
    const useRetrieval = this.elements.useRetrieval.value === 'false' ? false : this.elements.useRetrieval.value;
    const chunkFn = this.elements.chunkFn.value || 'chunk_by_section';
    const retrievalModel = this.elements.retrievalModel.value.trim() || 'all-MiniLM-L6-v2';
    
    return {
      task,
      doc_metadata: docMetadata,
      context: context ? [context] : [],
      max_rounds: maxRounds,
      logging_id: loggingId,
      max_jobs_per_round: maxJobsPerRound,
      num_tasks_per_round: numTasksPerRound,
      num_samples_per_task: numSamplesPerTask,
      use_retrieval: useRetrieval,
      chunk_fn: chunkFn,
      retrieval_model: retrievalModel
    };
  }

  async startMinions() {
    const formData = this.collectFormData();

    if (!formData.task) {
      this.logMessage('Task description is required', 'error');
      this.elements.task.focus();
      return;
    }

    // Disable start button and show loading state
    this.elements.startBtn.disabled = true;
    this.elements.startBtn.innerHTML = '<span class="spinner"></span>Running Minions Protocol...';

    const startTime = Date.now();

    try {
      this.logMessage('Starting advanced minions protocol...', 'info');
      this.logMessage(`Task: ${formData.task}`, 'info');
      this.logMessage(`Document Type: ${formData.doc_metadata}`, 'info');
      
      if (formData.context.length > 0) {
        this.logMessage(`Context length: ${formData.context[0].length} characters`, 'info');
      }
      
      this.logMessage(`Configuration: ${formData.max_rounds} rounds, ${formData.num_tasks_per_round} tasks/round`, 'debug');
      
      if (formData.use_retrieval && formData.use_retrieval !== 'false') {
        this.logMessage(`Using ${formData.use_retrieval} retrieval with ${formData.chunk_fn}`, 'debug');
      }

      const data = await this.makeRequest('/minions', {
        method: 'POST',
        body: JSON.stringify(formData)
      });

      const endTime = Date.now();
      const executionTime = (endTime - startTime) / 1000;

      this.logMessage('Minions protocol completed successfully!', 'success');
      this.logMessage('='.repeat(60), 'info');
      this.logMessage('FINAL ANSWER:', 'success');
      this.logMessage(data.final_answer, 'info');
      this.logMessage('='.repeat(60), 'info');

      // Display metrics
      this.displayMetrics({
        executionTime: data.execution_time || executionTime,
        remoteUsage: data.usage?.remote,
        localUsage: data.usage?.local,
        timing: data.timing,
        logFile: data.log_file,
        parametersUsed: data.parameters_used
      });

    } catch (error) {
      this.logMessage(`Minions execution failed: ${error.message}`, 'error');
      
      // Try to parse error details if available
      try {
        const errorData = JSON.parse(error.message);
        if (errorData.traceback) {
          this.logMessage('Error details:', 'error');
          this.logMessage(errorData.traceback, 'error');
        }
      } catch (e) {
        // Error message is not JSON, just log as is
      }
    } finally {
      this.elements.startBtn.disabled = false;
      this.elements.startBtn.innerHTML = '🚀 Start Minions Protocol';
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
      this.addMetric('Remote Total Tokens', metrics.remoteUsage.total_tokens || 'N/A', metricsElement);
      this.addMetric('Remote Prompt Tokens', metrics.remoteUsage.prompt_tokens || 'N/A', metricsElement);
      this.addMetric('Remote Completion Tokens', metrics.remoteUsage.completion_tokens || 'N/A', metricsElement);
    }
    
    // Local usage
    if (metrics.localUsage) {
      this.addMetric('Local Total Tokens', metrics.localUsage.total_tokens || 'N/A', metricsElement);
      this.addMetric('Local Prompt Tokens', metrics.localUsage.prompt_tokens || 'N/A', metricsElement);
      this.addMetric('Local Completion Tokens', metrics.localUsage.completion_tokens || 'N/A', metricsElement);
    }
    
    // Timing information
    if (metrics.timing) {
      Object.entries(metrics.timing).forEach(([key, value]) => {
        if (typeof value === 'number') {
          this.addMetric(
            key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()), 
            `${value.toFixed(2)}s`, 
            metricsElement
          );
        }
      });
    }
    
    // Parameters used
    if (metrics.parametersUsed) {
      this.addMetric('Max Rounds', metrics.parametersUsed.max_rounds || 'N/A', metricsElement);
      this.addMetric('Tasks/Round', metrics.parametersUsed.num_tasks_per_round || 'N/A', metricsElement);
      this.addMetric('Samples/Task', metrics.parametersUsed.num_samples_per_task || 'N/A', metricsElement);
      
      if (metrics.parametersUsed.use_retrieval && metrics.parametersUsed.use_retrieval !== 'false') {
        this.addMetric('Retrieval Method', metrics.parametersUsed.use_retrieval, metricsElement);
        this.addMetric('Chunk Function', metrics.parametersUsed.chunk_fn || 'N/A', metricsElement);
      }
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
  new MinionsAdvancedClient();
});

// Also initialize if the script is loaded after DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    new MinionsAdvancedClient();
  });
} else {
  new MinionsAdvancedClient();
}
