import React from 'react';

const LANGUAGE_OPTIONS = [
  'English',
  'Korean',
  'Spanish',
  'French',
  'German',
  'Japanese',
  'Chinese',
  'Hindi',
  'Arabic',
  'Portuguese'
];

const RecognitionDisplay = ({ data, liveSequence = [], settings, onSettingsChange }) => {
  const playAudio = (audioBase64) => {
    if (!audioBase64) return;
    const audio = new Audio(`data:audio/mp3;base64,${audioBase64}`);
    audio.play();
  };

  const handleToggle = (key) => (event) => {
    onSettingsChange({
      ...settings,
      [key]: event.target.checked
    });
  };

  const handleLanguageChange = (event) => {
    onSettingsChange({
      ...settings,
      targetLanguage: event.target.value
    });
  };

  const liveWords = liveSequence.length ? liveSequence.join(' ') : 'Waiting for signs…';

  return (
    <div className="recognition-display">
      <h2>Recognition Results</h2>

      <div className={`status-badge status-${data.status}`}>
        {data.status === 'idle' && '🟢 Ready'}
        {data.status === 'recognizing' && '🔵 Recognizing'}
        {data.status === 'recording_started' && '🔴 Recording'}
        {data.status === 'recording_ended' && '✅ Processed'}
      </div>

      <div className="settings-panel">
        <h3>LLM & Translation Settings</h3>
        <div className="toggle-row">
          <label className="toggle">
            <input
              type="checkbox"
              checked={settings.useLLM}
              onChange={handleToggle('useLLM')}
            />
            <span>Use LLM refinement</span>
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={settings.useTranslation}
              onChange={handleToggle('useTranslation')}
              disabled={!settings.useLLM}
            />
            <span>Translate output</span>
          </label>
        </div>
        <div className="select-row">
          <label className="select-field">
            Target language
            <select
              value={settings.targetLanguage}
              onChange={handleLanguageChange}
              disabled={!settings.useLLM || !settings.useTranslation}
            >
              {LANGUAGE_OPTIONS.map(language => (
                <option key={language} value={language}>
                  {language}
                </option>
              ))}
            </select>
          </label>
        </div>
        <p className="settings-note">
          These settings will apply once the LLM API is connected.
        </p>
      </div>

      <div className="recognized-text live-sequence">
        <h3>Live Words</h3>
        <div className="text-output">{liveWords}</div>
      </div>

      {data.emotion && (
        <div className="emotion-display">
          <h3>Your Emotion</h3>
          <p>
            {data.emotion === 'Happy' && '😊'}
            {data.emotion === 'Sad' && '😢'}
            {data.emotion === 'Angry' && '😠'}
            {data.emotion === 'Neutral' && '😐'}
            {(data.emotion === 'Surprise' || data.emotion === 'Surprised') && '😲'}
            {data.emotion === 'Fear' && '😨'}
            {data.emotion === 'Disgust' && '🤢'}
            {' '}{data.emotion}
            {data.emotion_confidence && 
              ` (${(data.emotion_confidence * 100).toFixed(1)}%)`
            }
          </p>
        </div>
      )}

      {data.detected_sign && (
        <div className="recognized-text">
          <h3>Recognized Sign</h3>
          <div className="text-output">
            <strong>{data.detected_sign}</strong>
            {data.confidence && (
              <>
                <div className="confidence-bar" style={{ marginTop: '0.5rem' }}>
                  <div
                    className="confidence-fill"
                    style={{ width: `${data.confidence * 100}%` }}
                  />
                </div>
                <p style={{ fontSize: '0.9rem', marginTop: '0.25rem' }}>
                  Confidence: {(data.confidence * 100).toFixed(1)}%
                </p>
              </>
            )}
          </div>
        </div>
      )}

      {data.recognized_sequence && (
        <div className="recognized-text">
          <h3>Full Sequence</h3>
          <div className="text-output">
            {data.recognized_sequence}
          </div>
        </div>
      )}

      {data.llm_response && (
        <div className="recognized-text">
          <h3>AI Response</h3>
          <div className="text-output">
            {data.llm_response}
          </div>
          {data.audio && (
            <button
              className="btn btn-primary"
              onClick={() => playAudio(data.audio)}
              style={{ marginTop: '0.5rem' }}
            >
              🔊 Play Audio
            </button>
          )}
        </div>
      )}
    </div>
  );
};

export default RecognitionDisplay;
