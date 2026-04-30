# Frontend Documentation

## Components

### App.jsx
Main application component with tab navigation.

### VideoFeed.jsx
- Captures video from webcam
- Sends frames to backend via WebSocket
- Displays real-time status

### RecognitionDisplay.jsx
- Shows recognized signs
- Displays emotion detection
- Plays audio responses

### VideoUpload.jsx
- File upload interface
- Progress tracking
- Batch processing

### LearningModule.jsx
- Interactive sign learning
- Visual guides
- Practice tips

## Services

### websocketService.js
Handles WebSocket connection and frame transmission.

### apiService.js
Handles HTTP requests to backend API.

## Styling

All styles in `App.css`:
- Gradient backgrounds
- Responsive grid layout
- Smooth animations
- Accessible color scheme

## Build

```bash
npm run build
```

Production build goes to `build/` directory.
