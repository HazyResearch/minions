import { Message, Usage, ProtocolCallback, ProtocolResult } from '../types/protocol';

export abstract class BaseProtocol {
  protected localClient: any; // Replace with proper client type
  protected remoteClient: any; // Replace with proper client type
  protected maxRounds: number;
  protected callback?: ProtocolCallback;
  protected logDir: string;

  constructor(
    localClient: any,
    remoteClient: any,
    maxRounds: number = 3,
    callback?: ProtocolCallback,
    logDir: string = 'minion_logs'
  ) {
    this.localClient = localClient;
    this.remoteClient = remoteClient;
    this.maxRounds = maxRounds;
    this.callback = callback;
    this.logDir = logDir;
  }

  protected async chatWithClient(
    client: any,
    messages: Message[],
    isLocal: boolean
  ): Promise<[string[], Usage]> {
    try {
      const [responses, usage] = await client.chat(messages);
      return [responses, usage];
    } catch (error) {
      console.error(`Error in ${isLocal ? 'local' : 'remote'} client:`, error);
      throw error;
    }
  }

  protected removeJsonBlocks(text: string): string {
    return text.replace(/```json\n[\s\S]*?\n```/g, '');
  }

  protected parseJson<T>(text: string): T {
    try {
      // First try to find JSON in markdown code blocks
      const blockMatches = text.match(/```(?:json)?\s*([\s\S]*?)```/g);
      if (blockMatches) {
        // Get the last code block
        const lastBlock = blockMatches[blockMatches.length - 1];
        const jsonStr = lastBlock.replace(/```(?:json)?\s*/, '').replace(/```$/, '').trim();
        return JSON.parse(jsonStr) as T;
      }

      // Then try to find JSON in curly braces
      const bracketMatches = text.match(/\{[\s\S]*?\}/g);
      if (bracketMatches) {
        // Get the last JSON object
        const lastBracket = bracketMatches[bracketMatches.length - 1];
        return JSON.parse(lastBracket) as T;
      }

      // Finally, try to parse the entire text as JSON
      return JSON.parse(text) as T;
    } catch (error) {
      console.error('Error parsing JSON:', error);
      console.error('Text that failed to parse:', text);
      throw error;
    }
  }

  abstract run(
    task: string,
    context: string[],
    maxRounds?: number,
    docMetadata?: string,
    loggingId?: string,
    isPrivacy?: boolean,
    images?: string[]
  ): Promise<ProtocolResult>;
} 