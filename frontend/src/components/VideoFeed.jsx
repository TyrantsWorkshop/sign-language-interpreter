import React, { useRef, useEffect, useState } from 'react';
import { websocketService } from '../services/websocketService';

const VideoFeed = ({ onRecognition }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isActive, setIsActive] = useState(false);
  const [fps, setFps] = useState(0);
  const frameCountRef = useRef(0);
  const lastTimeRef = useRef(Date.now());

  useEffect(() => {
    if (!isActive) return;

    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false
        });
        videoRef.current.srcObject = stream;
      } catch (error) {
        console.error('Camera error:', error);
        alert('Could not access camera. Please check permissions.');
        setIsActive(false);
      }
    };

    startCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
    };
  }, [isActive]);

  useEffect(() => {
    if (!isActive || !videoRef.current) return;

    const processFrame = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!videoRef.current?.readyState === videoRef.current?.HAVE_ENOUGH_DATA) {
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(blob => {
          websocketService.sendFrame(blob);
        });
      }

      // Calculate FPS
      frameCountRef.current++;
      const now = Date.now();
      if (now - lastTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastTimeRef.current = now;
      }

      requestAnimationFrame(processFrame);
    };

    const interval = setInterval(processFrame, 33); // ~30fps

    return () => clearInterval(interval);
  }, [isActive]);

  useEffect(() => {
    websocketService.onMessage = (data) => {
      onRecognition(data);
    };

    return () => {
      websocketService.onMessage = null;
    };
  }, [onRecognition]);

  const handleStart = async () => {
    setIsActive(true);
    await websocketService.connect();
  };

  const handleStop = () => {
    setIsActive(false);
    websocketService.disconnect();
  };

  return (
    <div className="video-feed-container">
      <div style={{ position: 'relative' }}>
        <video
          ref={videoRef}
          className="video-element"
          autoPlay
          playsInline
          style={{ display: isActive ? 'block' : 'none' }}
        />
        <canvas
          ref={canvasRef}
          style={{ display: 'none' }}
        />
        {!isActive && (
          <div style={{
            width: '100%',
            height: '400px',
            background: '#000',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#666',
            borderRadius: '8px'
          }}>
            Camera is off. Click Start to begin.
          </div>
        )}
        {isActive && (
          <div style={{
            position: 'absolute',
            top: '10px',
            right: '10px',
            background: 'rgba(0,0,0,0.7)',
            color: 'white',
            padding: '8px 12px',
            borderRadius: '4px',
            fontSize: '0.9rem'
          }}>
            FPS: {fps}
          </div>
        )}
      </div>
      <div className="controls" style={{ marginTop: '1.5rem' }}>
        {!isActive ? (
          <button className="btn btn-primary" onClick={handleStart}>
            ▶️ Start Recognition
          </button>
        ) : (
          <button className="btn btn-danger" onClick={handleStop}>
            ⏹️ Stop
          </button>
        )}
      </div>
    </div>
  );
};

export default VideoFeed;
