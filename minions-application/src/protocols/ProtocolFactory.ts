import { Config } from '../components/Sidebar';
import { BaseProtocol, ProtocolCallback } from '../types/protocol';
import { MinionProtocol } from './MinionProtocol';
import { MinionsProtocol } from './MinionsProtocol';

export class ProtocolFactory {
  static createProtocol(config: Config, callback?: ProtocolCallback): BaseProtocol {
    switch (config.protocol) {
      case 'Minion':
        return new MinionProtocol(config, callback);
      case 'Minions':
        return new MinionsProtocol(config, callback);
      case 'Minions-MCP':
        // TODO: Implement Minions-MCP protocol
        throw new Error('Minions-MCP protocol not implemented yet');
      default:
        throw new Error(`Unsupported protocol: ${config.protocol}`);
    }
  }
} 