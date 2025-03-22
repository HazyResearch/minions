import React from 'react';

const FileUploader: React.FC = () => {
  return (
    <div className="mb-4">
      <label htmlFor="fileUploader" className="block text-sm font-medium mb-1">
        Upload File(s)
      </label>
      <input type="file" id="fileUploader" className="w-full" multiple />
    </div>
  );
};

export default FileUploader;
