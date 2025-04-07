import { Message, Usage } from '../types/protocol';
import { BaseClient, TogetherClientConfig, ChatOptions } from './types';

export class TogetherClient implements BaseClient {
  private config: TogetherClientConfig;

  constructor(config: TogetherClientConfig) {
    this.config = config;
    console.log('Initializing TogetherClient with config:', {
      ...config,
      api_key: config.api_key ? '***' : undefined
    });
  }

  async chat(messages: Message[], options?: ChatOptions): Promise<[string[], Usage]> {
    console.log('TogetherClient.chat called with:', {
      messages,
      options,
      model: this.config.model_name
    });

    try {
      const response = await fetch('https://api.together.xyz/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.config.api_key}`,
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          model: this.config.model_name,
          messages: messages.map(msg => ({
            role: msg.role === 'assistant' ? 'assistant' : 'user',
            content: msg.content
          })),
          temperature: options?.temperature ?? this.config.temperature ?? 0.0,
          max_tokens: options?.max_tokens ?? this.config.max_tokens ?? 2048
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Together API error response:', {
          status: response.status,
          statusText: response.statusText,
          body: errorText
        });
        throw new Error(`Together API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const data = await response.json();
      console.log('Together API response:', data);

      return [
        [data.choices[0].message.content],
        {
          prompt_tokens: data.usage?.prompt_tokens ?? 0,
          completion_tokens: data.usage?.completion_tokens ?? 0,
          total_tokens: data.usage?.total_tokens ?? 0
        }
      ];
    } catch (error) {
      console.error('Together API error:', error);
      throw error;
    }
  }

  private formatMessages(messages: Message[]): string {
    return messages.map(msg => {
      const role = msg.role === 'assistant' ? 'Assistant' : 'Human';
      return `${role}: ${msg.content}`;
    }).join('\n');
  }
} 