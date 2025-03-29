export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface JobManifest {
  chunk: string;
  task: string;
  advice: string;
  chunk_id?: number;
  task_id?: number;
  job_id?: number;
}

export interface JobOutput {
  answer: string | null;
  explanation: string;
  citation: string | null;
}

export interface Job {
  manifest: JobManifest;
  output: JobOutput;
  sample: string;
  include?: boolean;
}

export interface ProtocolConfig {
  remoteProvider: 'OpenAI' | 'Anthropic' | 'Together';
  remoteApiKey: string;
  localProvider: 'Ollama';
  protocol: 'Minion' | 'Minions' | 'Minions-MCP';
  voiceEnabled: boolean;
}

export interface ProtocolResult {
  finalAnswer: string;
  meta?: {
    local?: {
      jobs: Job[];
    };
    remote?: {
      messages: any[];
    };
  }[];
  localUsage?: Usage;
  remoteUsage?: Usage;
  setupTime?: number;
  executionTime?: number;
  usage?: Usage;
}

export interface ProtocolOptions {
  contextDescription: string;
  fileMetadata: string;
  mcpToolsInfo?: string;
}

export type ProtocolCallback = (role: string, message: any, isFinal: boolean) => void;

export interface BaseProtocol {
  run(query: string, context: string[], options: ProtocolOptions): Promise<ProtocolResult>;
}

export interface ProtocolInput {
  query: string;
  context: string[];
  contextDescription?: string;
  fileMetadata?: any;
  config: {
    protocol: string;
    localProvider: string;
    remoteProvider: string;
    remoteApiKey: string;
    localModelName: string;
    remoteModelName: string;
    localTemperature: number;
    localMaxTokens: number;
    remoteTemperature: number;
    remoteMaxTokens: number;
    reasoningEffort: number;
    privacyMode: boolean;
    useBM25: boolean;
    images?: string[];
  };
}

export interface ProtocolOutput {
  output: string;
  loading: boolean;
  error: string | null;
  runProtocol: (input: ProtocolInput) => Promise<void>;
}

export interface ProtocolResponse {
  output: {
    final_answer: string;
    execution_time: number;
    setup_time: number;
    supervisor_messages?: Array<{
      role: string;
      content: string;
    }>;
    worker_messages?: Array<{
      role: string;
      content: string;
    }>;
    meta?: Array<{
      local?: { jobs: Job[] };
      remote?: { messages: string[] };
    }>;
    stats?: {
      total_local_jobs: number;
      total_remote_messages: number;
    };
    remote_usage?: Usage;
    local_usage?: Usage;
    log_file?: string;
  };
} 