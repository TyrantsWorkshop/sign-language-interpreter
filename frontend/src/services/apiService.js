const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const apiService = {
  async recognizeVideo(file, onProgress) {
    const formData = new FormData();
    formData.append('file', file);

    const xhr = new XMLHttpRequest();

    return new Promise((resolve, reject) => {
      xhr.upload.addEventListener('progress', (e) => {
        if (onProgress) {
          onProgress({ loaded: e.loaded, total: e.total });
        }
      });

      xhr.addEventListener('load', () => {
        if (xhr.status === 200) {
          resolve(JSON.parse(xhr.responseText));
        } else {
          reject(new Error('Upload failed'));
        }
      });

      xhr.addEventListener('error', () => {
        reject(new Error('Network error'));
      });

      xhr.open('POST', `${API_URL}/api/recognize-video`);
      xhr.send(formData);
    });
  },

  async healthCheck() {
    const response = await fetch(`${API_URL}/api/health`);
    return response.json();
  },

  async getModelInfo() {
    const response = await fetch(`${API_URL}/api/models/info`);
    return response.json();
  }
};
