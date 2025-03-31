import React from 'react';

const ProtocolSelector: React.FC = () => {
  return (
    <div className="mb-4">
      <label htmlFor="protocol" className="block text-sm font-medium mb-1">
        Protocol
      </label>
      <select id="protocol" className="w-full border border-gray-300 dark:border-gray-600 rounded p-2">
        <option value="minion">Minion</option>
        <option value="minions">Minions</option>
      </select>
    </div>
  );
};

export default ProtocolSelector;
