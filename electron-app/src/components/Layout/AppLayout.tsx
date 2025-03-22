import React, { useEffect, useState } from 'react';
import Sidebar from '../Sidebar/Sidebar.tsx';
import ContextInput from '../Main/ContextInput.tsx';
import QueryInput from '../Main/QueryInput.tsx';
import { useDarkMode } from '../../hooks/useDarkMode.ts';

const AppLayout: React.FC = () => {
    const [isDark, setIsDark] = useDarkMode();
  
    return (
      <div className="flex min-h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-white">
        <Sidebar />
        <main className="flex-1 p-6 space-y-6 overflow-y-auto">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold">Minions Protocol Playground</h1>
            <button
              className="text-sm px-3 py-1 rounded bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition"
              onClick={() => setIsDark(!isDark)}
            >
              Toggle {isDark ? 'Light' : 'Dark'} Mode
            </button>
          </div>
          <ContextInput />
          <QueryInput />
        </main>
      </div>
    );
  };
  
  export default AppLayout;