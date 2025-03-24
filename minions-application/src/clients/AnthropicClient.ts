import { Message, Usage } from '../types/protocol';
import { BaseClient, AnthropicClientConfig, ChatOptions } from './types';

export class AnthropicClient implements BaseClient {
  private config: AnthropicClientConfig;

  constructor(config: AnthropicClientConfig) {
    this.config = config;
    console.log('Initializing AnthropicClient with config:', {
      ...config,
      api_key: config.api_key ? '***' : undefined
    });
  }

  async chat(messages: Message[], options?: ChatOptions): Promise<[string[], Usage]> {
    console.log('AnthropicClient.chat called with:', {
      messages,
      options,
      model: this.config.model_name
    });

    try {
      const response = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-api-key': this.config.api_key,
          'anthropic-version': '2023-06-01',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          model: this.config.model_name,
          messages: messages.map(msg => ({
            role: msg.role === 'assistant' ? 'assistant' : 'user',
            content: msg.content
          })),
          max_tokens: options?.max_tokens ?? this.config.max_tokens ?? 4096,
          temperature: options?.temperature ?? this.config.temperature ?? 0.0
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Anthropic API error response:', {
          status: response.status,
          statusText: response.statusText,
          body: errorText
        });
        throw new Error(`Anthropic API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const data = await response.json();
      console.log('Anthropic API response:', data);

      return [
        [data.content[0].text],
        {
          prompt_tokens: data.usage?.input_tokens ?? 0,
          completion_tokens: data.usage?.output_tokens ?? 0,
          total_tokens: (data.usage?.input_tokens ?? 0) + (data.usage?.output_tokens ?? 0)
        }
      ];
    } catch (error) {
      console.error('Anthropic API error:', error);
      throw error;
    }
  }
} 