import { BaseProtocol, ProtocolResult, ProtocolOptions, ProtocolCallback } from '../types/protocol';

export abstract class BaseProtocolClass implements BaseProtocol {
  protected localClient: any;
  protected remoteClient: any;
  protected callback?: ProtocolCallback;
  protected maxRounds: number;

  constructor(localClient: any, remoteClient: any, maxRounds: number, callback?: ProtocolCallback) {
    this.localClient = localClient;
    this.remoteClient = remoteClient;
    this.callback = callback;
    this.maxRounds = maxRounds;
  }

  abstract run(query: string, context: string[], options: ProtocolOptions): Promise<ProtocolResult>;

  protected parseJson(text: string): any {
    try {
      // First try to find JSON in markdown code blocks
      const codeBlockRegex = /```(?:json)?\s*(\{[\s\S]*?\})\s*```/g;
      const matches = [...text.matchAll(codeBlockRegex)];
      if (matches.length > 0) {
        // Use the last code block
        return JSON.parse(matches[matches.length - 1][1]);
      }

      // Then try to find JSON objects in curly braces
      const jsonRegex = /\{[\s\S]*?\}/g;
      const jsonMatches = [...text.matchAll(jsonRegex)];
      if (jsonMatches.length > 0) {
        // Use the last JSON object
        return JSON.parse(jsonMatches[jsonMatches.length - 1][0]);
      }

      // Finally, try to parse the entire text as JSON
      return JSON.parse(text);
    } catch (error) {
      console.error('Failed to parse JSON:', error);
      console.error('Text that failed to parse:', text);
      throw error;
    }
  }
} 