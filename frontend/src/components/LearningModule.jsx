import React, { useState } from 'react';

const LearningModule = () => {
  const [selectedSign, setSelectedSign] = useState(null);

  const signs = [
    { name: 'HELLO', emoji: '👋', description: 'Wave your hand from side to side' },
    { name: 'GOODBYE', emoji: '👋', description: 'Wave your hand downward' },
    { name: 'THANK YOU', emoji: '🙏', description: 'Bring hand to mouth, move outward' },
    { name: 'LOVE', emoji: '❤️', description: 'Cross arms over chest' },
    { name: 'YES', emoji: '✅', description: 'Nod your head down repeatedly' },
    { name: 'NO', emoji: '❌', description: 'Move hand side to side' },
    { name: 'PLEASE', emoji: '🙏', description: 'Rub chest in circular motion' },
    { name: 'SORRY', emoji: '😔', description: 'Make fist, rub chest' },
  ];

  return (
    <div className="learning-container">
      <h2>Learn Sign Language</h2>
      <p style={{ color: '#666', marginBottom: '1.5rem' }}>
        Click on any sign to learn how to perform it correctly.
      </p>

      <div className="sign-grid">
        {signs.map((sign) => (
          <div
            key={sign.name}
            className="sign-card"
            onClick={() => setSelectedSign(sign)}
          >
            <div className="emoji">{sign.emoji}</div>
            <div className="name">{sign.name}</div>
          </div>
        ))}
      </div>

      {selectedSign && (
        <div style={{
          marginTop: '2rem',
          padding: '2rem',
          background: '#f5f7fa',
          borderRadius: '8px',
          border: '2px solid #667eea'
        }}>
          <button
            onClick={() => setSelectedSign(null)}
            style={{
              float: 'right',
              background: 'none',
              border: 'none',
              fontSize: '1.5rem',
              cursor: 'pointer'
            }}
          >
            ✕
          </button>
          <h3>{selectedSign.name}</h3>
          <div style={{ fontSize: '3rem', margin: '1rem 0' }}>
            {selectedSign.emoji}
          </div>
          <p style={{ fontSize: '1.1rem', color: '#667eea', marginTop: '1rem' }}>
            <strong>How to perform:</strong>
          </p>
          <p style={{ fontSize: '1rem', lineHeight: '1.6', marginTop: '0.5rem' }}>
            {selectedSign.description}
          </p>
          <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'white', borderRadius: '4px' }}>
            <p style={{ fontSize: '0.9rem', color: '#666' }}>
              💡 Tip: Practice in front of a mirror to see how your signing looks. 
              Make sure your hand movements are clear and deliberate.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default LearningModule;
