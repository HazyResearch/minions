import { Config } from '../components/Sidebar';
import { OllamaClient } from './OllamaClient';
import { OpenAIClient } from './OpenAIClient';
import { AnthropicClient } from './AnthropicClient';
import { TogetherClient } from './TogetherClient';

export class ClientFactory {
  static createClient(config: Config, isLocal: boolean): any {
    console.log('Creating client with config:', {
      ...config,
      remoteApiKey: config.remoteApiKey ? '***' : undefined
    });

    const provider = isLocal ? config.localProvider : config.remoteProvider;
    const modelName = isLocal ? config.localModelName : config.remoteModelName;
    const temperature = isLocal ? config.localTemperature : config.remoteTemperature;
    const maxTokens = isLocal ? config.localMaxTokens : config.remoteMaxTokens;

    console.log('Client parameters:', {
      provider,
      modelName,
      temperature,
      maxTokens,
      isLocal
    });

    switch (provider) {
      case 'Ollama':
        return new OllamaClient({
          model_name: modelName,
          temperature,
          max_tokens: maxTokens
        });
      case 'OpenAI':
        if (!config.remoteApiKey) {
          throw new Error('OpenAI API key is required');
        }
        return new OpenAIClient({
          api_key: config.remoteApiKey,
          model_name: modelName,
          temperature,
          max_tokens: maxTokens
        });
      case 'Anthropic':
        if (!config.remoteApiKey) {
          throw new Error('Anthropic API key is required');
        }
        return new AnthropicClient({
          api_key: config.remoteApiKey,
          model_name: modelName,
          temperature,
          max_tokens: maxTokens
        });
      case 'Together':
        if (!config.remoteApiKey) {
          throw new Error('Together API key is required');
        }
        // Map model names to Together.ai format
        const togetherModelMap: { [key: string]: string } = {
          'gpt-4': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
          'gpt-3.5-turbo': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
          'claude-3-opus': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
          'claude-3-sonnet': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
          'claude-3-haiku': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
          'DeepSeek-V3': 'deepseek-ai/DeepSeek-V3',
          'DeepSeek-R1': 'deepseek-ai/DeepSeek-R1',
          'Qwen2.5-72B': 'Qwen/Qwen2.5-72B-Instruct-Turbo',
          'Meta-Llama-3.1-405B': 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
          'Llama-3.3-70B': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
          'QWQ-32B': 'Qwen/QwQ-32B-Preview'
        };

        // Helper function to format model names
        const formatTogetherModelName = (name: string): string => {
          if (name in togetherModelMap) {
            return togetherModelMap[name];
          }
          if (name.includes('/')) {
            return name;
          }
          // Try to guess the organization based on the model name
          if (name.toLowerCase().startsWith('deepseek')) {
            return `deepseek-ai/${name}`;
          }
          if (name.toLowerCase().startsWith('llama')) {
            return `meta-llama/${name}`;
          }
          if (name.toLowerCase().startsWith('qwen')) {
            return `Qwen/${name}`;
          }
          console.warn(`Could not determine organization for model ${name}, using default`);
          return 'deepseek-ai/DeepSeek-V3';
        };

        const togetherModel = formatTogetherModelName(modelName);
        console.log('Using Together model:', togetherModel);
        return new TogetherClient({
          api_key: config.remoteApiKey,
          model_name: togetherModel,
          temperature,
          max_tokens: maxTokens
        });
      default:
        throw new Error(`Unsupported provider: ${provider}`);
    }
  }
} 