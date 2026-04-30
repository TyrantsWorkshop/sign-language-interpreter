const WS_URL = (process.env.REACT_APP_WS_URL || 'ws://localhost:8000').replace('http', 'ws');

export const websocketService = {
  ws: null,
  onMessage: null,
  messageQueue: [],

  async connect() {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(`${WS_URL}/ws/sign-language`);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          // Send queued messages
          this.messageQueue.forEach(msg => this.ws.send(msg));
          this.messageQueue = [];
          resolve();
        };

        this.ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (this.onMessage) {
            this.onMessage(data);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          reject(error);
        };

        this.ws.onclose = () => {
          console.log('WebSocket closed');
        };
      } catch (error) {
        reject(error);
      }
    });
  },

  sendFrame(blob) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(blob);
    } else {
      this.messageQueue.push(blob);
    }
  },

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
};
