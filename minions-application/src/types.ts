import { Config } from './components/Sidebar';

export interface Message {
  role: string;
  content: string;
  is_final: boolean;
}

export interface ProtocolInput {
  query: string;
  context: string[];
  config: Config;
  contextDescription: string;
  fileMetadata: string;
}

export interface ProtocolOutput {
  output: string;
  loading: boolean;
  streamingMessages: Message[];
  runProtocol: (input: ProtocolInput) => Promise<void>;
}

export interface MinionResponse {
  output: {
    final_answer: string;
    supervisor_messages: Array<{ role: string; content: string }>;
    worker_messages: Array<{ role: string; content: string }>;
    remote_usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
    local_usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
    log_file: string;
    setup_time: number;
    execution_time: number;
  };
}

export interface MinionsResponse {
  output: {
    final_answer: string;
    meta: Array<{
      local?: { jobs: Array<any> };
      remote?: { messages: Array<any> };
    }>;
    stats?: {
      total_local_jobs: number;
      total_remote_messages: number;
    };
    setup_time: number;
    execution_time: number;
  };
}

export type ProtocolResponse = MinionResponse | MinionsResponse; 