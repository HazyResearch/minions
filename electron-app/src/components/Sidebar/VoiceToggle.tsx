const VoiceToggle = () => (
  <div>
    <label className="block text-sm font-medium mb-1">Voice Mode</label>
    <div className="flex items-center gap-2">
      <input type="checkbox" className="rounded text-blue-600" />
      <span className="text-sm">Enable Minion Voice</span>
    </div>
  </div>
);

export default VoiceToggle;
