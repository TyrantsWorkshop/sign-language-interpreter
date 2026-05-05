import React from 'react';
import './App.css';
import VideoFeed from './components/VideoFeed';
import RecognitionDisplay from './components/RecognitionDisplay';
import LearningModule from './components/LearningModule';
import VideoUpload from './components/VideoUpload';

function App() {
  const [activeTab, setActiveTab] = React.useState('live');
  const [recognitionData, setRecognitionData] = React.useState({
    status: 'idle',
    detected_sign: null,
    confidence: 0,
    emotion: 'Neutral',
    gesture: 'RECORDING',
    llm_response: null,
    audio: null
  });
  const [liveSequence, setLiveSequence] = React.useState([]);
  const [settings, setSettings] = React.useState({
    useLLM: true,
    useTranslation: false,
    targetLanguage: 'English'
  });

  const handleRecognition = (data) => {
    setRecognitionData(prev => ({ ...prev, ...data }));

    if (data.status === 'recording_started') {
      setLiveSequence([]);
    }

    if (data.status === 'recognizing' && data.detected_sign) {
      setLiveSequence(prev => {
        const lastWord = prev[prev.length - 1];
        if (lastWord === data.detected_sign) {
          return prev;
        }
        return [...prev, data.detected_sign];
      });
    }

    if (data.status === 'recording_ended' && data.recognized_sequence) {
      const words = data.recognized_sequence.split(' ').filter(Boolean);
      setLiveSequence(words);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🤟 Sign Language Interpreter</h1>
        <p>Real-time ASL Recognition with AI</p>
      </header>

      <div className="nav-tabs">
        <button 
          className={`tab ${activeTab === 'live' ? 'active' : ''}`}
          onClick={() => setActiveTab('live')}
        >
          🎥 Live Recognition
        </button>
        <button 
          className={`tab ${activeTab === 'upload' ? 'active' : ''}`}
          onClick={() => setActiveTab('upload')}
        >
          📹 Upload Video
        </button>
        <button 
          className={`tab ${activeTab === 'learn' ? 'active' : ''}`}
          onClick={() => setActiveTab('learn')}
        >
          📚 Learning Module
        </button>
      </div>

      <div className="content">
        {activeTab === 'live' && (
          <div className="live-section">
            <VideoFeed onRecognition={handleRecognition} />
            <RecognitionDisplay
              data={recognitionData}
              liveSequence={liveSequence}
              settings={settings}
              onSettingsChange={setSettings}
            />
          </div>
        )}
        
        {activeTab === 'upload' && (
          <VideoUpload onRecognition={handleRecognition} />
        )}
        
        {activeTab === 'learn' && (
          <LearningModule />
        )}
      </div>

      <footer className="app-footer">
        <p>Built with 💙 for accessibility | Sign Language Interpreter v1.0</p>
      </footer>
    </div>
  );
}

export default App;
