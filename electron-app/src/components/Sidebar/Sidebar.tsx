import React from 'react';
import ProviderSettings from './ProviderSettings.tsx';
import ModelSelector from './ModelSelector.tsx';
import ProtocolSelector from './ProtocolSelector.tsx';
import VoiceToggle from './VoiceToggle.tsx';

const Sidebar: React.FC = () => {
  return (
    <aside className="w-64 bg-white dark:bg-gray-800 p-4 border-r border-gray-300 dark:border-gray-700">
      <h2 className="text-xl font-bold mb-4">Settings</h2>
      <ProviderSettings />
      <ModelSelector />
      <ProtocolSelector />
      <VoiceToggle />
    </aside>
  );
};

export default Sidebar;
