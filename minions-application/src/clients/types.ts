import { Message, Usage } from '../types/protocol';

export interface ChatOptions {
  response_format?: { type: string };
  temperature?: number;
  max_tokens?: number;
  model?: string;
}

export interface ChatResponse {
  messages: string[];
  usage: Usage;
  done_reasons?: string[];
}

export interface BaseClient {
  chat: (messages: Message[], options?: ChatOptions) => Promise<[string[], Usage]>;
}

export interface OllamaClientConfig {
  model_name: string;
  base_url?: string;
  temperature?: number;
  max_tokens?: number;
}

export interface OpenAIClientConfig {
  api_key: string;
  model_name: string;
  temperature?: number;
  max_tokens?: number;
}

export interface AnthropicClientConfig {
  api_key: string;
  model_name: string;
  temperature?: number;
  max_tokens?: number;
}

export interface TogetherClientConfig {
  api_key: string;
  model_name: string;
  temperature?: number;
  max_tokens?: number;
}

export type ClientConfig = 
  | OllamaClientConfig 
  | OpenAIClientConfig 
  | AnthropicClientConfig 
  | TogetherClientConfig; 