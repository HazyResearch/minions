import React from 'react';

const ProviderSettings: React.FC = () => {
  return (
    <div className="mb-4">
      <label htmlFor="provider" className="block text-sm font-medium mb-1">
        LLM Provider
      </label>
      <select id="provider" className="w-full border border-gray-300 dark:border-gray-600 rounded p-2">
        <option value="openai">OpenAI</option>
        <option value="together">Together</option>
      </select>
    </div>
  );
};

export default ProviderSettings;
