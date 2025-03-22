import React from 'react';

const WebScraperInput: React.FC = () => {
  return (
    <div className="mb-4">
      <label htmlFor="urlInput" className="block text-sm font-medium mb-1">
        URL for Scraping
      </label>
      <input
        type="url"
        id="urlInput"
        className="w-full border border-gray-300 dark:border-gray-600 rounded p-2"
        placeholder="Enter URL..."
      />
    </div>
  );
};

export default WebScraperInput;
