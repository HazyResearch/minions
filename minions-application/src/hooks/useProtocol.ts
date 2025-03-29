import { useState } from 'react';
import { ProtocolInput, ProtocolOutput, ProtocolResponse } from '../types/protocol';

const API_BASE_URL = 'http://127.0.0.1:8080/api';

export function useProtocol(): ProtocolOutput {
  const [output, setOutput] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const runProtocol = async (input: ProtocolInput): Promise<void> => {
    // reset states
    setOutput('');
    setError(null);
    setLoading(true);

    try {
      const requestBody = {
        task: input.query,
        context: input.context.join('\n'), 
        doc_metadata: input.fileMetadata || '',
        protocol: input.config.protocol,
        localProvider: input.config.localProvider,
        remoteProvider: input.config.remoteProvider,
        remoteApiKey: input.config.remoteApiKey,
        localModelName: input.config.localModelName,
        remoteModelName: input.config.remoteModelName,
        localTemperature: input.config.localTemperature,
        localMaxTokens: input.config.localMaxTokens,
        remoteTemperature: input.config.remoteTemperature,
        remoteMaxTokens: input.config.remoteMaxTokens,
        reasoningEffort: input.config.reasoningEffort,
        privacy_mode: input.config.privacyMode,
        use_bm25: input.config.useBM25,
        images: input.config.images
      };

      console.log('Sending request:', {
        ...requestBody,
        remote_api_key: '***', 
        context: requestBody.context.slice(0, 300) + '...',
      });

      const response = await fetch(`${API_BASE_URL}/run_protocol`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        let errorMessage = 'Failed to run protocol';
        try {
          const errorData = await response.json();
          console.error('Server error response:', errorData);
          errorMessage = errorData.error || errorMessage;
        } catch (e) {
          console.error('Failed to parse error response:', e);
          errorMessage = response.statusText;
        }
        throw new Error(errorMessage);
      }

      const data = await response.json() as ProtocolResponse;
      console.log('Protocol completed with result:', data);

      let formattedOutput = `Final Answer: ${data.output.final_answer}\n`;

      if (data.output.execution_time > 0) {
        formattedOutput += `\nExecution Time: ${data.output.execution_time.toFixed(2)}s`;
      }

      if (data.output.meta) {
        data.output.meta.forEach((round, index) => {
          if (round.remote?.messages) {
            formattedOutput += `\n\nRound ${index + 1} Remote Messages:`;
            round.remote.messages.forEach(msg => {
              try {
                const jsonMsg = typeof msg === 'string' ? JSON.parse(msg) : msg;
                formattedOutput += `\n${JSON.stringify(jsonMsg, null, 2)}`;
              } catch {
                formattedOutput += `\n${msg}`;
              }
            });
          }
        });
      }

      setOutput(formattedOutput);
    } catch (error) {
      console.error('Protocol error:', error);
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          setError('Request timed out after 5 minutes. Please try again.');
        } else {
          setError(error.message);
        }
      } else {
        setError('An unexpected error occurred');
      }
    } finally {
      setLoading(false);
    }
  };

  return {
    output,
    loading,
    error,
    runProtocol
  };
}

export default useProtocol;