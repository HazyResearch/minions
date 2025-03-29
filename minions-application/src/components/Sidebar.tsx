import React from 'react';

export interface Config {
  remoteProvider: 'OpenAI' | 'Anthropic' | 'Together';
  remoteApiKey: string;
  localProvider: 'Ollama';
  protocol: 'Minion' | 'Minions' | 'Minions-MCP';
  voiceEnabled: boolean;
  privacyMode: boolean;
  useBM25: boolean;
  localModelName: string;
  remoteModelName: string;
  localTemperature: number;
  localMaxTokens: number;
  remoteTemperature: number;
  remoteMaxTokens: number;
  reasoningEffort: 'low' | 'medium' | 'high';
  images?: string[];  // Optional array of base64-encoded images
}

interface SidebarProps {
  config: Config;
  setConfig: (config: Config) => void;
}

const Sidebar: React.FC<SidebarProps> = ({ config, setConfig }) => {
  const handleConfigChange = (key: keyof Config, value: any) => {
    setConfig({ ...config, [key]: value });
  };

  return (
    <div className="w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 h-screen overflow-y-auto">
      <div className="p-6 space-y-8">
        <div className="flex items-center space-x-3 mb-8">
          <svg className="w-8 h-8 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          <h2 className="text-xl font-bold text-gray-800 dark:text-gray-200">Settings</h2>
        </div>

        <div className="space-y-6">
          {/* Provider Settings */}
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-xl">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Provider Settings</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Remote Provider</label>
                <select
                  value={config.remoteProvider}
                  onChange={(e) => handleConfigChange('remoteProvider', e.target.value)}
                  className="w-full p-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                >
                  <option value="OpenAI">OpenAI</option>
                  <option value="Anthropic">Anthropic</option>
                  <option value="Together">Together</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Remote API Key</label>
                <input
                  type="password"
                  value={config.remoteApiKey}
                  onChange={(e) => handleConfigChange('remoteApiKey', e.target.value)}
                  className="w-full p-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Local Provider</label>
                <select
                  value={config.localProvider}
                  onChange={(e) => handleConfigChange('localProvider', e.target.value)}
                  className="w-full p-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                >
                  <option value="Ollama">Ollama</option>
                </select>
              </div>
            </div>
          </div>

          {/* Model Settings */}
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-xl">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Model Settings</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Protocol</label>
                <select
                  value={config.protocol}
                  onChange={(e) => handleConfigChange('protocol', e.target.value)}
                  className="w-full p-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                >
                  <option value="Minion">Minion</option>
                  <option value="Minions">Minions</option>
                  <option value="Minions-MCP">Minions-MCP</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Local Model Name</label>
                <input
                  type="text"
                  value={config.localModelName}
                  onChange={(e) => handleConfigChange('localModelName', e.target.value)}
                  className="w-full p-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Remote Model Name</label>
                <input
                  type="text"
                  value={config.remoteModelName}
                  onChange={(e) => handleConfigChange('remoteModelName', e.target.value)}
                  className="w-full p-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                />
              </div>
            </div>
          </div>

          {/* Temperature Settings */}
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-xl">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Temperature Settings</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Local Temperature</label>
                <input
                  type="number"
                  value={config.localTemperature}
                  onChange={(e) => handleConfigChange('localTemperature', parseFloat(e.target.value))}
                  step="0.1"
                  min="0"
                  max="1"
                  className="w-full p-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Remote Temperature</label>
                <input
                  type="number"
                  value={config.remoteTemperature}
                  onChange={(e) => handleConfigChange('remoteTemperature', parseFloat(e.target.value))}
                  step="0.1"
                  min="0"
                  max="1"
                  className="w-full p-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                />
              </div>
            </div>
          </div>

          {/* Token Settings */}
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-xl">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Token Settings</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Local Max Tokens</label>
                <input
                  type="number"
                  value={config.localMaxTokens}
                  onChange={(e) => handleConfigChange('localMaxTokens', parseInt(e.target.value))}
                  min="1"
                  max="4096"
                  className="w-full p-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Remote Max Tokens</label>
                <input
                  type="number"
                  value={config.remoteMaxTokens}
                  onChange={(e) => handleConfigChange('remoteMaxTokens', parseInt(e.target.value))}
                  min="1"
                  max="4096"
                  className="w-full p-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                />
              </div>
            </div>
          </div>

          {/* Reasoning Settings */}
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-xl">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Reasoning Settings</h3>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Reasoning Effort</label>
                <select
                  value={config.reasoningEffort}
                  onChange={(e) => handleConfigChange('reasoningEffort', e.target.value)}
                  className="w-full p-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-200"
                >
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
            </div>
          </div>

          {/* Feature Toggles */}
          <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-xl">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-4">Feature Toggles</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Enable Voice</label>
                <div className="relative inline-flex h-6 w-11 items-center rounded-full bg-gray-200 dark:bg-gray-600">
                  <input
                    type="checkbox"
                    checked={config.voiceEnabled}
                    onChange={(e) => handleConfigChange('voiceEnabled', e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="peer h-4 w-4 rounded-full bg-white transition-all duration-200 peer-checked:translate-x-full peer-checked:bg-blue-500"></div>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Privacy Mode</label>
                <div className="relative inline-flex h-6 w-11 items-center rounded-full bg-gray-200 dark:bg-gray-600">
                  <input
                    type="checkbox"
                    checked={config.privacyMode}
                    onChange={(e) => handleConfigChange('privacyMode', e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="peer h-4 w-4 rounded-full bg-white transition-all duration-200 peer-checked:translate-x-full peer-checked:bg-blue-500"></div>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300">Use BM25 (Minions only)</label>
                <div className="relative inline-flex h-6 w-11 items-center rounded-full bg-gray-200 dark:bg-gray-600">
                  <input
                    type="checkbox"
                    checked={config.useBM25}
                    onChange={(e) => handleConfigChange('useBM25', e.target.checked)}
                    className="sr-only peer"
                  />
                  <div className="peer h-4 w-4 rounded-full bg-white transition-all duration-200 peer-checked:translate-x-full peer-checked:bg-blue-500"></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;
