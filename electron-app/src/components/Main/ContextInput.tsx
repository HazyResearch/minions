import { useState } from 'react';

export default function ContextInput({
  onContextChange,
}: {
  onContextChange?: (value: string) => void;
}) {
  const [context, setContext] = useState('');

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = e.target.value;
    setContext(value);
    if (onContextChange) onContextChange(value);
  };

  return (
    <div className="mb-6">
      <label className="block text-sm font-medium mb-1">Context</label>
      <textarea
        placeholder="Paste your document context here..."
        value={context}
        onChange={handleChange}
        className="w-full p-3 border border-gray-300 dark:border-gray-700 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white h-32"
      />
    </div>
  );
}
