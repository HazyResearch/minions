import { Config } from '../components/Sidebar';
import { ProtocolResult, ProtocolOptions, ProtocolCallback} from '../types/protocol';
import { ClientFactory } from '../clients/ClientFactory';
import { BaseProtocolClass } from './BaseProtocolClass';

export class MinionsProtocol extends BaseProtocolClass {
  constructor(config: Config, callback?: ProtocolCallback) {
    super(ClientFactory.createClient(config, true), ClientFactory.createClient(config, false), 5, callback);
  }

  async run(
    task: string,
    context: string[],
    options: ProtocolOptions = {
      contextDescription: "",
      fileMetadata: "",
      mcpToolsInfo: ""
    }
  ): Promise<ProtocolResult> {
    try {
      // Make API call to server
      const response = await fetch('/api/run_protocol', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          task,
          context,
          protocol: 'Minions',
          doc_metadata: options.fileMetadata,
          use_bm25: false,
          // Add any other necessary parameters from the config
          localModelName: this.localClient.modelName,
          remoteModelName: this.remoteClient.modelName,
          localProvider: this.localClient.provider,
          remoteProvider: this.remoteClient.provider,
          localTemperature: this.localClient.temperature,
          remoteTemperature: this.remoteClient.temperature,
          localMaxTokens: this.localClient.maxTokens,
          remoteMaxTokens: this.remoteClient.maxTokens,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      // Call the callback with progress updates if available
      if (this.callback && data.output.meta) {
        for (const roundMeta of data.output.meta) {
          if (roundMeta.local?.jobs) {
            this.callback('worker', roundMeta.local.jobs, true);
          }
          if (roundMeta.remote?.messages) {
            for (const message of roundMeta.remote.messages) {
              this.callback('supervisor', message, true);
            }
          }
        }
      }

      return {
        finalAnswer: data.output.final_answer,
        meta: data.output.meta,
        localUsage: data.output.local_usage,
        remoteUsage: data.output.remote_usage,
        setupTime: data.output.setup_time,
        executionTime: data.output.execution_time,
      };
    } catch (error) {
      console.error('Error running Minions protocol:', error);
      throw error;
    }
  }

} 