import React from 'react';

const ModelSelector: React.FC = () => {
  return (
    <div className="mb-4">
      <label htmlFor="localModel" className="block text-sm font-medium mb-1">
        Local Model
      </label>
      <select id="localModel" className="w-full border border-gray-300 dark:border-gray-600 rounded p-2">
        <option value="llama3.2">llama3.2</option>
        <option value="mistral">mistral</option>
      </select>
      <label htmlFor="remoteModel" className="block text-sm font-medium mt-2 mb-1">
        Remote Model
      </label>
      <select id="remoteModel" className="w-full border border-gray-300 dark:border-gray-600 rounded p-2">
        <option value="gpt-4o">gpt-4o</option>
        <option value="claude">claude</option>
      </select>
    </div>
  );
};

export default ModelSelector;
