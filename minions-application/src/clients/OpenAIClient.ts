import { Message, Usage } from '../types/protocol';
import { BaseClient, OpenAIClientConfig, ChatOptions } from './types';

export class OpenAIClient implements BaseClient {
  private config: OpenAIClientConfig;

  constructor(config: OpenAIClientConfig) {
    this.config = config;
    console.log('Initializing OpenAIClient with config:', {
      ...config,
      api_key: config.api_key ? '***' : undefined
    });
  }

  async chat(messages: Message[], options?: ChatOptions): Promise<[string[], Usage]> {
    console.log('OpenAIClient.chat called with:', {
      messages,
      options,
      model: this.config.model_name
    });

    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
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
          max_tokens: options?.max_tokens ?? this.config.max_tokens ?? 4096
        })
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('OpenAI API error response:', {
          status: response.status,
          statusText: response.statusText,
          body: errorText
        });
        throw new Error(`OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const data = await response.json();
      console.log('OpenAI API response:', data);

      return [
        [data.choices[0].message.content],
        {
          prompt_tokens: data.usage?.prompt_tokens ?? 0,
          completion_tokens: data.usage?.completion_tokens ?? 0,
          total_tokens: data.usage?.total_tokens ?? 0
        }
      ];
    } catch (error) {
      console.error('OpenAI API error:', error);
      throw error;
    }
  }
} 