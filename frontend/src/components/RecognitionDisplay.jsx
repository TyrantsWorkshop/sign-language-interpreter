import React from 'react';

const RecognitionDisplay = ({ data }) => {
  const playAudio = (audioBase64) => {
    if (!audioBase64) return;
    const audio = new Audio(`data:audio/mp3;base64,${audioBase64}`);
    audio.play();
  };

  return (
    <div className="recognition-display">
      <h2>Recognition Results</h2>

      <div className={`status-badge status-${data.status}`}>
        {data.status === 'idle' && '🟢 Ready'}
        {data.status === 'recognizing' && '🔵 Recognizing'}
        {data.status === 'recording_started' && '🔴 Recording'}
        {data.status === 'recording_ended' && '✅ Processed'}
      </div>

      {data.emotion && (
        <div className="emotion-display">
          <h3>Your Emotion</h3>
          <p>
            {data.emotion === 'Happy' && '😊'}
            {data.emotion === 'Sad' && '😢'}
            {data.emotion === 'Angry' && '😠'}
            {data.emotion === 'Neutral' && '😐'}
            {data.emotion === 'Surprised' && '😲'}
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
