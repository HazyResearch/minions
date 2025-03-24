import React, { useState, useEffect, useRef } from 'react';

interface Message {
  role: 'user' | 'supervisor' | 'worker' | 'assistant';
  content: string;
  timestamp: Date;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface ChatInterfaceProps {
  query: string;
  setQuery: (query: string) => void;
  output: string;
  runTask: (query: string) => void;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ query, setQuery, output, runTask }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [finalAnswer, setFinalAnswer] = useState<string | React.ReactNode | null>(null);
  const [executionTime, setExecutionTime] = useState<number | null>(null);
  const [lastSubmittedQuery, setLastSubmittedQuery] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (output) {
      console.log('ChatInterface received new output:', output);
      
      const lines = output.split('\n');
      const newMessages: Message[] = [];
      
      lines.forEach(line => {
        console.log('Processing line:', line);
        if (line.startsWith('Supervisor: ')) {
          newMessages.push({
            role: 'supervisor',
            content: line.slice(11),
            timestamp: new Date()
          });
        } else if (line.startsWith('Worker: ')) {
          newMessages.push({
            role: 'worker',
            content: line.slice(8),
            timestamp: new Date()
          });
        } else if (line.startsWith('Final Answer: ')) {
          const answer = line.slice(13);
          try {
            // Try to parse as JSON if it's a string
            const jsonAnswer = typeof answer === 'string' ? JSON.parse(answer) : answer;
            // Always stringify the answer to ensure proper display
            setFinalAnswer(
              <pre className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg overflow-x-auto">
                {JSON.stringify(jsonAnswer, null, 2)}
              </pre>
            );
          } catch {
            // If not JSON or parsing fails, use as is
            setFinalAnswer(answer);
          }
        } else if (line.startsWith('Execution Time: ')) {
          const time = parseFloat(line.slice(15));
          setExecutionTime(time);
        }
      });

      console.log('Adding new messages:', newMessages);
      setMessages(prev => {
        const updated = [...prev, ...newMessages];
        console.log('Updated messages:', updated);
        return updated;
      });
    }
  }, [output]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      console.log('Submitting query:', query);
      setIsProcessing(true);
      setLastSubmittedQuery(query);
      setMessages(prev => {
        const updated = [...prev, {
          role: 'user' as const,
          content: query,
          timestamp: new Date()
        }];
        console.log('Updated messages after user input:', updated);
        return updated;
      });
      
      try {
        console.log('Running task...');
        await runTask(query);
        console.log('Task completed');
      } catch (error) {
        console.error('Task failed:', error);
      } finally {
        setIsProcessing(false);
        setQuery('');
        inputRef.current?.focus();
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const getMessageStyle = (role: Message['role']) => {
    switch (role) {
      case 'user':
        return 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-md';
      case 'supervisor':
        return 'bg-gradient-to-r from-purple-500 to-purple-600 text-white shadow-md';
      case 'worker':
        return 'bg-gradient-to-r from-green-500 to-green-600 text-white shadow-md';
      case 'assistant':
        return 'bg-gradient-to-r from-yellow-500 to-yellow-600 text-white shadow-md';
      default:
        return 'bg-gradient-to-r from-gray-500 to-gray-600 text-white shadow-md';
    }
  };

  const getRoleLabel = (role: Message['role']) => {
    switch (role) {
      case 'user':
        return 'You';
      case 'supervisor':
        return 'Remote Model';
      case 'worker':
        return 'Local Model';
      case 'assistant':
        return 'Final Answer';
      default:
        return role;
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 h-full flex flex-col">
      <div className="p-3 flex flex-col h-full">
        <div className="flex items-center space-x-2 mb-3">
          <svg className="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
          </svg>
          <h2 className="text-base font-semibold text-gray-900 dark:text-gray-100">Chat</h2>
        </div>

        <div className="flex flex-col flex-1 min-h-0 space-y-3">
          <div className="flex flex-col space-y-1">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Query</label>
            <div className="flex space-x-2">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && runTask(query)}
                placeholder="Enter your query..."
                className="flex-1 p-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200 text-sm"
              />
              <button
                onClick={() => runTask(query)}
                className="px-3 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors duration-200 flex items-center justify-center space-x-2 text-sm"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                <span>Run</span>
              </button>
            </div>
          </div>

          <div className="flex flex-col flex-1 min-h-0 space-y-1">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Output</label>
            <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-3 flex-1 overflow-y-auto">
              <pre className="whitespace-pre-wrap text-sm text-gray-700 dark:text-gray-300 font-mono">
                {output.split('\n').map((line, index) => {
                  if (line.startsWith('Final Answer: ')) {
                    const answer = line.slice(13);
                    try {
                      // Try to parse as JSON if it's a string
                      const jsonAnswer = typeof answer === 'string' ? JSON.parse(answer) : answer;
                      // Always stringify the answer to ensure proper display
                      return `Final Answer: ${JSON.stringify(jsonAnswer, null, 2)}`;
                    } catch {
                      // If not JSON or parsing fails, use as is
                      return line;
                    }
                  }
                  return line;
                }).join('\n')}
              </pre>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
