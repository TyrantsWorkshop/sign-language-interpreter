import React, { useState } from 'react';
import { apiService } from '../services/apiService';

const VideoUpload = ({ onRecognition }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [dragover, setDragover] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragover(true);
  };

  const handleDragLeave = () => {
    setDragover(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragover(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      setFile(files[0]);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      alert('Please select a video file');
      return;
    }

    setLoading(true);
    setProgress(0);

    try {
      const result = await apiService.recognizeVideo(file, (progressEvent) => {
        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );
        setProgress(percentCompleted);
      });

      onRecognition({
        status: 'recording_ended',
        recognized_sequence: result.recognized_sequence,
        llm_response: result.llm_response,
        emotion: 'Mixed',
        timestamp: result.timestamp
      });
    } catch (error) {
      console.error('Upload error:', error);
      alert('Error processing video: ' + error.message);
    } finally {
      setLoading(false);
      setProgress(0);
    }
  };

  return (
    <div className="upload-container">
      <h2>Upload Sign Language Video</h2>
      <p style={{ color: '#666', marginBottom: '1.5rem' }}>
        Upload a video file (MP4, WebM) to get recognized signs and AI-generated responses.
      </p>

      <div
        className={`upload-area ${dragover ? 'dragover' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="upload-icon">📁</div>
        <p style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>
          Drag and drop your video here
        </p>
        <p style={{ color: '#999', marginBottom: '1rem' }}>
          or click to browse
        </p>
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          style={{ display: 'none' }}
          id="file-input"
        />
        <label htmlFor="file-input" className="btn btn-primary">
          Choose File
        </label>
      </div>

      {file && (
        <div style={{ marginTop: '1.5rem' }}>
          <p>
            <strong>Selected file:</strong> {file.name}
          </p>
          <p style={{ color: '#666', fontSize: '0.9rem' }}>
            Size: {(file.size / 1024 / 1024).toFixed(2)} MB
          </p>
        </div>
      )}

      {progress > 0 && (
        <div className="progress" style={{ marginTop: '1rem' }}>
          <div className="progress-bar" style={{ width: `${progress}%` }} />
        </div>
      )}

      <button
        className="btn btn-primary"
        onClick={handleUpload}
        disabled={!file || loading}
        style={{ marginTop: '1.5rem', width: '100%' }}
      >
        {loading ? '⏳ Processing...' : '🚀 Recognize Signing'}
      </button>
    </div>
  );
};

export default VideoUpload;
