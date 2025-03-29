import React, { useState } from 'react';
import Sidebar, { Config } from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import ContextInput from './components/ContextInput';
import useProtocol from './hooks/useProtocol';

const App: React.FC = () => {
  const [config, setConfig] = useState<Config>({
    remoteProvider: 'OpenAI',
    remoteApiKey: '',
    localProvider: 'Ollama',
    protocol: 'Minion',
    voiceEnabled: false,
    privacyMode: false,
    useBM25: false,
    localModelName: 'llama3.2',
    remoteModelName: 'gpt-4',
    localTemperature: 0.0,
    localMaxTokens: 2048,
    remoteTemperature: 0.2,
    remoteMaxTokens: 2048,
    reasoningEffort: 'medium',
    images: []
  });
  const [context, setContext] = useState<string[]>([]);
  const [contextDescription, setContextDescription] = useState('');
  const [fileMetadata, setFileMetadata] = useState('');
  const [query, setQuery] = useState('');
  const [error, setError] = useState<string | null>(null);
  const { output, loading, runProtocol } = useProtocol();

  const handleRunTask = async (userQuery: string) => {
    setError(null);
    
    // validate configuration
    if (!config.remoteApiKey) {
      setError('Please enter a valid API key for the remote provider');
      return;
    }

    if (context.length === 0) {
      setError('Please provide some context for the query');
      return;
    }

    if (!contextDescription.trim()) {
      setError('Please provide a description of the context');
      return;
    }

    if (!userQuery.trim()) {
      setError('Please enter a query');
      return;
    }

    try {
      const reasoningEffortMap = {
        'low': 0,
        'medium': 1,
        'high': 2
      };
      
      await runProtocol({ 
        query: userQuery, 
        context, 
        config: {
          ...config,
          reasoningEffort: reasoningEffortMap[config.reasoningEffort]
        },
        contextDescription,
        fileMetadata
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred while running the protocol');
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="flex h-screen overflow-hidden">
        {/* Sidebar */}
        <div className="w-80 flex-shrink-0 overflow-y-auto border-r border-gray-200 dark:border-gray-700">
          <div className="p-4">
            <Sidebar
              config={config}
              setConfig={setConfig}
            />
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4">
            <div className="max-w-4xl mx-auto w-full space-y-4">
              <ContextInput
                context={context}
                setContext={setContext}
                contextDescription={contextDescription}
                setContextDescription={setContextDescription}
                setFileMetadata={setFileMetadata}
              />
              {error && (
                <div className="p-4 bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-100 rounded-lg">
                  {error}
                </div>
              )}
              <ChatInterface
                query={query}
                setQuery={setQuery}
                output={loading ? 'Loading...' : output}
                runTask={handleRunTask}
              />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
