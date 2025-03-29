import { Message, Usage } from '../types/protocol';
import { BaseClient, OllamaClientConfig, ChatOptions } from './types';

export class OllamaClient implements BaseClient {
  private config: OllamaClientConfig;
  private baseUrl: string;

  constructor(config: OllamaClientConfig) {
    this.config = config;
    this.baseUrl = config.base_url || 'http://localhost:11434';
  }

  async chat(messages: Message[], options?: ChatOptions): Promise<[string[], Usage]> {
    const formattedMessages = messages.map(msg => ({
      role: msg.role,
      content: msg.content,
      images: msg.images,
    }));

    const response = await fetch(`${this.baseUrl}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: this.config.model_name,
        messages: formattedMessages,
        stream: false,
        options: {
          temperature: options?.temperature || this.config.temperature || 0.7,
          num_predict: options?.max_tokens || this.config.max_tokens || 2048,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.statusText}`);
    }

    const data = await response.json();
    
    // Ollama doesn't provide token counts, so we'll estimate
    const estimatedUsage: Usage = {
      prompt_tokens: this.estimateTokens(messages.map(m => m.content).join(' ')),
      completion_tokens: this.estimateTokens(data.message.content),
      total_tokens: this.estimateTokens(messages.map(m => m.content).join(' ') + data.message.content),
    };

    return [[data.message.content], estimatedUsage];
  }

  private estimateTokens(text: string): number {
    // Rough estimation: 1 token ≈ 4 characters
    return Math.ceil(text.length / 4);
  }
} 