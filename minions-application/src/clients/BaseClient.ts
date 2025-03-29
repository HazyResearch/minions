import { Usage } from '../types/protocol';

export interface BaseClient {
    modelName: string;
    temperature: number;
    maxTokens: number;
    chat(messages: Array<{ role: string; content: string }>): Promise<[string[], Usage]>;
} 