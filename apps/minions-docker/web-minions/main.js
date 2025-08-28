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
            
            // PDF upload elements
            pdfSelectBtn: document.getElementById('pdf-select-btn'),
            pdfFileInput: document.getElementById('pdf-file-input'),
            uploadProgress: document.getElementById('upload-progress'),
            progressFill: document.getElementById('progress-fill'),
            progressText: document.getElementById('progress-text'),
            pdfInfo: document.getElementById('pdf-info'),
            pdfFilename: document.getElementById('pdf-filename'),
            pdfStats: document.getElementById('pdf-stats'),
            removePdf: document.getElementById('remove-pdf'),
            
            // Processing mode
            processingModeMinions: document.getElementById('mode_minions'),
            processingModeRemote: document.getElementById('mode_remote'),
            
            // Basic configuration
            loggingId: document.getElementById('logging_id'),
            
            // Buttons
            checkStatusBtn: document.getElementById('check_status'),
            startBtn: document.getElementById('start'),
            clearLogBtn: document.getElementById('clear_log'),
            
            // Output
            log: document.getElementById('log'),
            metricsContainer: document.getElementById('metrics-container'),
            metrics: document.getElementById('metrics')
        };

        // Initialize PDF upload state
        this.pdfData = {
            filename: null,
            text: null,
            pages: 0,
            characters: 0
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
    
    
    // Update start button state when task or context changes
    this.elements.task.oninput = () => this.updateStartButtonState();
    this.elements.context.oninput = () => this.updateStartButtonState();
    
    // Auto-generate logging ID when task changes
    this.elements.task.onblur = () => this.autoGenerateLoggingId();

    // Processing mode change listeners
    this.elements.processingModeMinions.onchange = () => this.updateProcessingMode();
    this.elements.processingModeRemote.onchange = () => this.updateProcessingMode();

    // PDF upload event listeners
    this.attachPdfEventListeners();
    
    // Initialize processing mode
    this.updateProcessingMode();
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
    const context = this.elements.context.value.trim();
    const hasValidInput = task && context;
    
    this.elements.startBtn.disabled = !hasValidInput;
    
    // Update button text to indicate what's missing
    if (!hasValidInput) {
      const isMinionsMode = this.elements.processingModeMinions.checked;
      if (!task && !context) {
        this.elements.startBtn.innerHTML = isMinionsMode ? 
          '🚀 Start Minions Protocol (Task & Context Required)' : 
          '☁️ Start Remote Processing (Task & Context Required)';
      } else if (!task) {
        this.elements.startBtn.innerHTML = isMinionsMode ? 
          '🚀 Start Minions Protocol (Task Required)' : 
          '☁️ Start Remote Processing (Task Required)';
      } else if (!context) {
        this.elements.startBtn.innerHTML = isMinionsMode ? 
          '🚀 Start Minions Protocol (Context Required)' : 
          '☁️ Start Remote Processing (Context Required)';
      }
    } else {
      // Reset to normal text when everything is valid
      this.updateProcessingMode();
    }
  }

  updateProcessingMode() {
    const isMinionsMode = this.elements.processingModeMinions.checked;
    const startBtn = this.elements.startBtn;
    
    if (isMinionsMode) {
      startBtn.innerHTML = '🚀 Start Minions Protocol';
    } else {
      startBtn.innerHTML = '☁️ Start Remote Processing';
    }
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
    const loggingId = this.elements.loggingId.value.trim() || null;

    return {
      task,
      doc_metadata: docMetadata,
      context: context ? [context] : [],
      logging_id: loggingId
    };
  }

  validateFormData(formData) {
    const errors = [];

    if (!formData.task) {
      errors.push('Task description is required');
    }

    if (!formData.context || formData.context.length === 0 || !formData.context[0]) {
      errors.push('Context is required. Please upload a PDF document or provide text content.');
    }

    return errors;
  }

  async startMinions() {
    const formData = this.collectFormData();
    const isMinionsMode = this.elements.processingModeMinions.checked;

    // Validate form data
    const validationErrors = this.validateFormData(formData);
    if (validationErrors.length > 0) {
      validationErrors.forEach(error => {
        this.logMessage(error, 'error');
      });
      
      // Focus on the appropriate field
      if (!formData.task) {
        this.elements.task.focus();
      } else if (!formData.context || formData.context.length === 0 || !formData.context[0]) {
        this.elements.context.focus();
      }
      return;
    }

    // Disable start button and show loading state
    this.elements.startBtn.disabled = true;
    const loadingText = isMinionsMode ? 
      '<span class="spinner"></span>Running Minions Protocol...' : 
      '<span class="spinner"></span>Processing with Remote Model...';
    this.elements.startBtn.innerHTML = loadingText;

    const startTime = Date.now();

    try {
      if (isMinionsMode) {
        this.logMessage('Starting advanced minions protocol...', 'info');
        this.logMessage(`Mode: Minions Protocol (cost-effective multi-agent processing)`, 'info');
      } else {
        this.logMessage('Starting remote model processing...', 'info');
        this.logMessage(`Mode: Remote Model Only (direct processing, higher cost)`, 'warning');
      }
      
      this.logMessage(`Task: ${formData.task}`, 'info');
      this.logMessage(`Document Type: ${formData.doc_metadata}`, 'info');
      
      if (formData.context.length > 0) {
        this.logMessage(`Context length: ${formData.context[0].length} characters`, 'info');
      }

      // Choose endpoint based on processing mode
      const endpoint = isMinionsMode ? '/minions' : '/remote-only';
      
      const data = await this.makeRequest(endpoint, {
        method: 'POST',
        body: JSON.stringify(formData)
      });

      const endTime = Date.now();
      const executionTime = (endTime - startTime) / 1000;

      const successMessage = isMinionsMode ? 
        'Minions protocol completed successfully!' : 
        'Remote model processing completed!';
      
      this.logMessage(successMessage, 'success');
      this.logMessage('='.repeat(60), 'info');
      this.logMessage('FINAL ANSWER:', 'success');
      this.logMessage(data.final_answer, 'info');
      this.logMessage('='.repeat(60), 'info');

      // Display metrics with processing mode context
      this.displayMetrics({
        executionTime: data.execution_time || executionTime,
        remoteUsage: data.usage?.remote,
        localUsage: data.usage?.local,
        timing: data.timing,
        logFile: data.log_file,
        parametersUsed: data.parameters_used,
        processingMode: isMinionsMode ? 'Minions Protocol' : 'Remote Only'
      });

    } catch (error) {
      const errorMessage = isMinionsMode ? 
        `Minions execution failed: ${error.message}` : 
        `Remote processing failed: ${error.message}`;
      
      this.logMessage(errorMessage, 'error');
      
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
      this.updateProcessingMode(); // This will set the correct button text
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

  // PDF Upload Methods
  attachPdfEventListeners() {
    const selectBtn = this.elements.pdfSelectBtn;
    const fileInput = this.elements.pdfFileInput;
    const removePdf = this.elements.removePdf;

    // Click button to select file
    selectBtn.onclick = () => {
      fileInput.click();
    };

    // File input change
    fileInput.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        this.handlePdfFile(file);
      }
    };

    // Remove PDF button
    removePdf.onclick = (e) => {
      e.stopPropagation();
      this.removePdf();
    };
  }

  async handlePdfFile(file) {
    // Validate file
    if (!file.type.includes('pdf') && !file.name.toLowerCase().endsWith('.pdf')) {
      this.logMessage('Please select a PDF file', 'error');
      return;
    }

    // Check file size (10MB limit)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      this.logMessage('PDF file must be smaller than 10MB', 'error');
      return;
    }

    this.logMessage(`Processing PDF: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`, 'info');

    // Show progress
    this.showPdfProgress();

    try {
      // Upload PDF to backend
      const formData = new FormData();
      formData.append('pdf', file);

      const response = await fetch(`${this.backendUrl}/upload-pdf`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || `HTTP ${response.status}`);
      }

      const data = await response.json();

      // Store PDF data
      this.pdfData = {
        filename: data.filename,
        text: data.text,
        pages: data.pages,
        characters: data.characters
      };

      // Update context with PDF text
      this.elements.context.value = data.text;

      // Update document metadata if it's still default
      if (this.elements.docMetadata.value === 'Document') {
        this.elements.docMetadata.value = 'PDF Document';
      }

      // Show PDF info
      this.showPdfInfo();

      this.logMessage(`PDF processed successfully: ${data.pages} pages, ${data.characters} characters extracted`, 'success');
      
      // Update button state since context is now populated
      this.updateStartButtonState();

    } catch (error) {
      this.logMessage(`PDF processing failed: ${error.message}`, 'error');
      this.hidePdfProgress();
    }
  }

  showPdfProgress() {
    // Hide select container and PDF info
    const selectContainer = document.querySelector('.pdf-select-container');
    if (selectContainer) selectContainer.style.display = 'none';
    this.elements.pdfInfo.style.display = 'none';
    
    // Show progress
    this.elements.uploadProgress.style.display = 'flex';
    
    // Animate progress bar
    let progress = 0;
    const progressInterval = setInterval(() => {
      progress += Math.random() * 15;
      if (progress > 90) progress = 90;
      
      this.elements.progressFill.style.width = `${progress}%`;
      
      if (progress > 50) {
        this.elements.progressText.textContent = 'Extracting text from PDF...';
      }
    }, 200);

    // Store interval for cleanup
    this.progressInterval = progressInterval;
  }

  hidePdfProgress() {
    // Clear progress interval
    if (this.progressInterval) {
      clearInterval(this.progressInterval);
      this.progressInterval = null;
    }

    // Hide progress
    this.elements.uploadProgress.style.display = 'none';
    
    // Show select container
    const selectContainer = document.querySelector('.pdf-select-container');
    if (selectContainer) selectContainer.style.display = 'block';
  }

  showPdfInfo() {
    // Clear progress interval and complete progress bar
    if (this.progressInterval) {
      clearInterval(this.progressInterval);
      this.progressInterval = null;
    }

    this.elements.progressFill.style.width = '100%';
    this.elements.progressText.textContent = 'PDF processed successfully!';

    // Wait a moment then show PDF info
    setTimeout(() => {
      this.elements.uploadProgress.style.display = 'none';
      const selectContainer = document.querySelector('.pdf-select-container');
      if (selectContainer) selectContainer.style.display = 'none';
      
      // Update PDF info
      this.elements.pdfFilename.textContent = this.pdfData.filename;
      this.elements.pdfStats.textContent = `${this.pdfData.pages} pages • ${this.pdfData.characters.toLocaleString()} characters`;
      
      // Show PDF info
      this.elements.pdfInfo.style.display = 'block';
    }, 1000);
  }

  removePdf() {
    // Store current PDF text before clearing
    const currentPdfText = this.pdfData.text;

    // Clear PDF data
    this.pdfData = {
      filename: null,
      text: null,
      pages: 0,
      characters: 0
    };

    // Clear context if it contains PDF text
    if (currentPdfText && this.elements.context.value === currentPdfText) {
      this.elements.context.value = '';
    }

    // Reset file input
    this.elements.pdfFileInput.value = '';

    // Hide PDF info and show select container
    this.elements.pdfInfo.style.display = 'none';
    this.elements.uploadProgress.style.display = 'none';
    const selectContainer = document.querySelector('.pdf-select-container');
    if (selectContainer) selectContainer.style.display = 'block';

    this.logMessage('PDF removed', 'info');
    
    // Update button state since context may now be empty
    this.updateStartButtonState();
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
