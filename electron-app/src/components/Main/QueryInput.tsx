import { useState } from 'react';
import axios from 'axios';

export default function QueryInput() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    setLoading(true);
    try { //hardcoded for now
      const res = await axios.post('http://localhost:5000/query', {
        query,
        context: '', 
        provider: 'OpenAI', 
        local_provider: 'Ollama',
        remote_model: 'gpt-4o',
        local_model: 'llama3.2',
        local_temperature: 0.0,
        local_max_tokens: 4096,
        remote_temperature: 0.0,
        remote_max_tokens: 4096,
        protocol: 'Minion',
        api_key: process.env.OPENAI_API_KEY, 
        metadata: 'Electron version test',
        images: null,
      });

      setResponse(res.data.final_answer || res.data.error);
    } catch (err: any) {
      console.error(err);
      setResponse('Something went wrong.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <label className="block text-sm font-medium mb-1">Query</label>
      <textarea
        placeholder="Enter your query here..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        className="w-full p-3 border border-gray-300 dark:border-gray-700 rounded bg-white dark:bg-gray-800 text-gray-900 dark:text-white h-24"
      />
      <button
        onClick={handleSubmit}
        className="mt-3 bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded disabled:opacity-50"
        disabled={loading || !query.trim()}
      >
        {loading ? 'Thinking...' : 'Submit'}
      </button>

      {response && (
        <div className="mt-4 p-4 rounded bg-gray-100 dark:bg-gray-700 text-black dark:text-white whitespace-pre-wrap">
          {response}
        </div>
      )}
    </div>
  );
}
