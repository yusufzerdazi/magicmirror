import React, { useState, useEffect, useRef } from 'react';
import { createFullEndpoint } from '#root/utils/apiUtils.ts';
import useConditionalAuth from '#root/src/hooks/useConditionalAuth';
import { IS_WARP_LOCAL } from '#root/utils/constants.ts';

const FRAME_WIDTH = 512;
const FRAME_HEIGHT = 512;
const INITIAL_PROMPT = "a mischievous cat with a third eye, matte pastel colour pallete in a cartoon style";
const INITIAL_RETRY_DELAY = 1000;
const MAX_RETRY_DELAY = 30000;
const BACKOFF_FACTOR = 1.5;
const FRAME_INTERVAL = 250; // Send 4 frames per second
const MAX_BUFFER_SIZE = 16; // Don't let buffer grow too large
const DISPLAY_DURATION = 0; // Time to show each frame before transition
const AUDIO_INTERVAL = 5000; // Process audio every 5 seconds
const MAX_TRANSCRIPT_LENGTH = 200; // Maximum characters to show
const SCROLL_DURATION = 5; // Seconds to keep text visible
const COIN_SIZE = 100; // Size in pixels
const ROTATION_DURATION = 3; // Seconds for one full rotation
const LEFT_PATH = "M -600 300 C -500 350 -400 250 -300 300 C -250 350 -200 250 -150 300 C -100 350 -50 250 0 350";  // Left wobbly curve
const RIGHT_PATH = "M 0 350 C 50 250 100 350 150 300 C 200 350 250 250 300 300 C 350 350 400 250 600 300";  // Right wobbly curve starting from center

type TranscriptEntry = {
  text: string;
  timestamp: number;
};

type DeviceInfo = {
  deviceId: string;
  label: string;
};

type Rotation = 0 | 90 | 180 | 270;

const buildWebsocketUrlFromPodId = (podId: string) => {
  return `ws://192.168.1.113:8765`;
};

const buildPromptEndpointUrlFromPodId = (podId: string) => {
  return `http://192.168.1.113:5556/prompt/`;
};

const WarpPage = () => {
  const { getToken } = useConditionalAuth();
  const [currentStream, setCurrentStream] = useState<MediaStream | null>(null);
  const [warp, setWarp] = useState<any>(null);
  const [isRendering, setIsRendering] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const croppedCanvasRef = useRef<HTMLCanvasElement>(null);
  const currentCanvasRef = useRef<HTMLCanvasElement>(null);
  const nextCanvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const frameQueueRef = useRef<HTMLImageElement[]>([]);
  const isTransitioningRef = useRef(false);
  const lastTransitionTime = useRef(Date.now());
  const [wsStatus, setWsStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const [transcription, setTranscription] = useState<string>('');
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([]);
  const [videoDevices, setVideoDevices] = useState<DeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [rotation, setRotation] = useState<Rotation>(0);
  const [totalTranscripts, setTotalTranscripts] = useState<number>(0);

  const addTranscript = (text: string) => {
    setTotalTranscripts(prev => prev + 1);
    setTranscripts(prev => {
      const now = Date.now();
      const newTranscripts = [...prev, { text, timestamp: now, id: totalTranscripts }];
      return newTranscripts.filter(t => now - t.timestamp < SCROLL_DURATION * 1000);
    });
  };

  const getVideoDevices = async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoDevices = devices
        .filter(device => device.kind === 'videoinput')
        .map(device => ({
          deviceId: device.deviceId,
          label: device.label || `Camera ${device.deviceId}`
        }));
      setVideoDevices(videoDevices);
      
      if (videoDevices.length > 0 && !selectedDeviceId) {
        setSelectedDeviceId(videoDevices[0].deviceId);
      }
    } catch (error) {
      console.error('Error getting video devices:', error);
    }
  };

  useEffect(() => {
    getVideoDevices();
    
    navigator.mediaDevices.addEventListener('devicechange', getVideoDevices);
    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', getVideoDevices);
    };
  }, []);

  useEffect(() => {
    const initializeWebcam = async () => {
      if (!selectedDeviceId) return;
      
      try {
        currentStream?.getTracks().forEach(track => track.stop());
        
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: {
            deviceId: { exact: selectedDeviceId },
          }
        });
        setCurrentStream(stream);
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setIsRendering(true);
      } catch (error) {
        console.error('Error initializing webcam:', error);
      }
    };

    initializeWebcam();
    return () => {
      currentStream?.getTracks().forEach(track => track.stop());
    };
  }, [selectedDeviceId]);

  // Send initial prompt when warp is ready
  useEffect(() => {
    if (warp?.podId && warp.podStatus === 'RUNNING') {
      const promptEndpointUrl = buildPromptEndpointUrlFromPodId(warp.podId);
      const encodedPrompt = encodeURIComponent(INITIAL_PROMPT);
      const endpoint = `${promptEndpointUrl}${encodedPrompt}`;

      fetch(endpoint, {
        method: 'POST',
      }).catch(error => {
        console.error('Error sending initial prompt:', error);
      });
    }
  }, [warp?.podId, warp?.podStatus]);

  // Initialize warp
  useEffect(() => {
    const initializeWarp = async () => {
      if (!getToken) return;
      
      const token = await getToken();
      const response = await fetch(createFullEndpoint(`warps`), {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const { entities } = await response.json();
        const warp = entities?.warps?.[0];
        if (warp?.podId) {
          setWarp(warp);
        }
      }
    };

    if (!IS_WARP_LOCAL) {
      initializeWarp();
    } else {
      setWarp({ id: 'local', podId: 'local', podStatus: 'RUNNING' });
    }
  }, [getToken]);

  // Frame display logic
  useEffect(() => {
    if (!isRendering) return;

    const displayNextFrame = () => {
      const now = Date.now();
      const timeSinceLastTransition = now - lastTransitionTime.current;

      // Check if it's time for next frame and we have frames to show
      if (!isTransitioningRef.current && 
          frameQueueRef.current.length > 0 && 
          timeSinceLastTransition >= DISPLAY_DURATION) {

        // Skip frames if queue is getting too large
        if (frameQueueRef.current.length > MAX_BUFFER_SIZE / 2) {
          // Keep the most recent frames and discard older ones
          const framesToSkip = Math.floor(frameQueueRef.current.length / 2);
          frameQueueRef.current = frameQueueRef.current.slice(framesToSkip);
        }

        const nextFrame = frameQueueRef.current[0];
        
        // Prepare next frame
        const nextCanvas = nextCanvasRef.current;
        const nextCtx = nextCanvas?.getContext('2d');
        
        if (nextFrame && nextCtx && nextCanvas) {
          // Draw next frame on bottom canvas
          nextCtx.clearRect(0, 0, FRAME_WIDTH, FRAME_HEIGHT);
          nextCtx.drawImage(nextFrame, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
          
          // Start transition
          isTransitioningRef.current = true;
          let opacity = 1;
          
          const fade = () => {
            if (opacity > 0 && currentCanvasRef.current) {
              opacity -= 0.05; // 20 steps for smooth fade
              currentCanvasRef.current.style.opacity = opacity.toString();
              requestAnimationFrame(fade);
            } else {
              // Transition complete - swap canvases and update z-indices
              const temp = currentCanvasRef.current;
              currentCanvasRef.current = nextCanvasRef.current;
              nextCanvasRef.current = temp;

              // Update z-indices to keep current canvas on top
              if (currentCanvasRef.current) {
                currentCanvasRef.current.style.zIndex = '2';
                currentCanvasRef.current.style.opacity = '1';
              }
              if (nextCanvasRef.current) {
                nextCanvasRef.current.style.zIndex = '1';
                nextCanvasRef.current.style.opacity = '1';
              }
              
              // Remove displayed frame from queue
              frameQueueRef.current = frameQueueRef.current.slice(1);
              
              lastTransitionTime.current = Date.now();
              isTransitioningRef.current = false;
            }
          };

          requestAnimationFrame(fade);
        }
      }

      requestAnimationFrame(displayNextFrame);
    };

    displayNextFrame();
  }, [isRendering]);

  // WebSocket connection
  useEffect(() => {
    if (!warp?.podId || warp.podStatus !== 'RUNNING') return;

    let retryCount = 0;
    let retryDelay = INITIAL_RETRY_DELAY;

    const connectWebSocket = () => {
      setWsStatus('connecting');
      const websocketUrl = buildWebsocketUrlFromPodId(warp.podId);
      const socket = new WebSocket(websocketUrl);
      socket.binaryType = 'arraybuffer';

      socket.onopen = () => {
        setWsStatus('connected');
        console.log('WebSocket connected');
        retryCount = 0;
        retryDelay = INITIAL_RETRY_DELAY;
      };

      socket.onmessage = event => {
        const blob = new Blob([event.data], { type: 'image/jpeg' });
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          URL.revokeObjectURL(url);
          // Only add to queue if we haven't exceeded MAX_BUFFER_SIZE
          if (frameQueueRef.current.length < MAX_BUFFER_SIZE) {
            frameQueueRef.current = [...frameQueueRef.current, img];
          }
        };
        img.src = url;
      };

      socket.onclose = () => {
        setWsStatus('disconnected');
        console.log(`WebSocket disconnected (attempt ${retryCount + 1})`);
        
        if (reconnectTimeoutRef.current) {
          clearTimeout(reconnectTimeoutRef.current);
        }

        // Calculate next retry delay with exponential backoff
        retryDelay = Math.min(retryDelay * BACKOFF_FACTOR, MAX_RETRY_DELAY);
        retryCount++;
        
        reconnectTimeoutRef.current = setTimeout(() => {
          if (socketRef.current?.readyState === WebSocket.CLOSED) {
            connectWebSocket();
          }
        }, retryDelay);
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      socketRef.current = socket;
    };

    connectWebSocket();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [warp?.podId, warp?.podStatus]);

  // Send frames
  useEffect(() => {
    if (!currentStream || !socketRef.current || wsStatus !== 'connected') return;

    const videoTrack = currentStream.getVideoTracks()?.[0];
    if (!videoTrack) return;

    const croppedCanvas = croppedCanvasRef.current;
    if (!croppedCanvas) return;

    const croppedCtx = croppedCanvas.getContext('2d');
    if (!croppedCtx) return;

    let frameInterval: NodeJS.Timeout;
    let lastFrameTime = 0;

    const sendFrame = async () => {
      const now = Date.now();
      // Ensure we maintain consistent timing
      if (now - lastFrameTime < FRAME_INTERVAL) {
        return;
      }

      if (videoRef.current && wsStatus === 'connected') {
        croppedCtx.drawImage(videoRef.current, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
        
        croppedCanvas.toBlob(
          blob => {
            if (blob && socketRef?.current?.readyState === WebSocket.OPEN) {
              blob.arrayBuffer().then(buffer => {
                socketRef?.current?.send(buffer);
                lastFrameTime = now;
              });
            }
          },
          'image/jpeg',
          0.8,
        );
      }
    };

    // Use a shorter interval to ensure smooth timing
    frameInterval = setInterval(sendFrame, FRAME_INTERVAL / 2);

    // Handle pings from server
    const handlePing = () => {
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        socketRef.current.send('pong');
      }
    };

    socketRef.current.addEventListener('ping', handlePing);

    return () => {
      clearInterval(frameInterval);
      socketRef.current?.removeEventListener('ping', handlePing);
    };
  }, [currentStream, wsStatus]);

  // Handle audio recording and processing
  useEffect(() => {
    if (!warp?.podId || warp.podStatus !== 'RUNNING') return;

    const initAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          }
        });

        // Add this before the MediaRecorder initialization to debug supported formats
        const debugSupportedMimeTypes = () => {
          const types = [
            'audio/webm',
            'audio/webm;codecs=opus',
            'audio/ogg;codecs=opus',
            'audio/mp4',
            'audio/mpeg',
            'audio/wav'
          ];
          
          console.log('Supported audio MIME types:');
          types.forEach(type => {
            console.log(`${type}: ${MediaRecorder.isTypeSupported(type)}`);
          });
        };

        // Call this function before creating the MediaRecorder
        debugSupportedMimeTypes();

        let selectedMimeType = '';
        for (const mimeType of ['audio/webm', 'audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4']) {
          if (MediaRecorder.isTypeSupported(mimeType)) {
            selectedMimeType = mimeType;
            break;
          }
        }

        if (!selectedMimeType) {
          throw new Error('No supported MIME type found for MediaRecorder');
        }

        const mediaRecorder = new MediaRecorder(stream, {
          mimeType: selectedMimeType
        });
        mediaRecorderRef.current = mediaRecorder;

        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          if (audioChunksRef.current.length === 0) return;

          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          audioChunksRef.current = [];

          const formData = new FormData();
          formData.append('audio', audioBlob);

          try {
            const response = await fetch(`http://192.168.1.113:5556/transcribe`, {
              method: 'POST',
              body: formData,
            });

            if (response.ok) {
              const data = await response.json();
              if (data.text && data.text.trim()) {
                const text = data.text.trim();
                console.log('🎤 Received transcription:', text);
                setTranscription(text);
                addTranscript(text);
              }
            } else {
              console.error('❌ Transcription failed:', await response.text());
            }
          } catch (error) {
            console.error('❌ Error processing audio:', error);
          }
        };

        mediaRecorder.onerror = (event) => {
          console.error('MediaRecorder error:', event);
          // Optionally attempt to restart recording
          if (mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
          }
        };

        // Improved recording cycle
        const startRecording = () => {
          if (mediaRecorder.state === 'inactive') {
            audioChunksRef.current = [];
            mediaRecorder.start();
          }
        };

        const stopRecording = () => {
          if (mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
          }
        };

        const recordingInterval = setInterval(() => {
          stopRecording();
          startRecording();
        }, AUDIO_INTERVAL);

        startRecording();

        return () => {
          clearInterval(recordingInterval);
          stopRecording();
          stream.getTracks().forEach(track => track.stop());
        };
      } catch (error) {
        console.error('Error initializing audio:', error);
        // Optionally show user-friendly error message
        if (error instanceof DOMException && error.name === 'NotAllowedError') {
          console.log('Microphone access was denied by the user');
        } else if (error instanceof DOMException && error.name === 'NotFoundError') {
          console.log('No microphone was found');
        }
      }
    };

    initAudio();
  }, [warp?.podId, warp?.podStatus]);

  useEffect(() => {
    const cleanup = setInterval(() => {
      setTranscripts(prev => {
        const now = Date.now();
        return prev.filter(t => now - t.timestamp < SCROLL_DURATION * 1000);
      });
    }, 16); // Run at ~60fps

    return () => clearInterval(cleanup);
  }, []);

  useEffect(() => {
    let animationFrame: number;
    
    const updateTranscripts = () => {
      setTranscripts(prev => {
        const now = Date.now();
        return prev.filter(t => now - t.timestamp < SCROLL_DURATION * 1000);
      });
      animationFrame = requestAnimationFrame(updateTranscripts);
    };

    animationFrame = requestAnimationFrame(updateTranscripts);
    
    return () => {
      cancelAnimationFrame(animationFrame);
    };
  }, []);

  const applyRotationStyle = (rotation: number) => {
    let transform = `rotate(${rotation}deg)`;
    let scale = rotation === 90 || rotation === 270 ? 'scale(calc(9/16))' : 'scale(1)';
    return `${transform} scale(1)`;
  };

  // Add this effect near the top of the component to inject the keyframes
  useEffect(() => {
    const styleSheet = document.createElement("style");
    styleSheet.textContent = `
      @keyframes coin-rotate {
        0% {
          transform: rotateY(0deg) rotateX(20deg);
        }
        100% {
          transform: rotateY(360deg) rotateX(20deg);
        }
      }
    `;
    document.head.appendChild(styleSheet);
    return () => {
      document.head.removeChild(styleSheet);
    };
  }, []);

  const SvgPaths = () => (
    <svg 
      className="fixed inset-0 w-screen h-screen pointer-events-none" 
      style={{ zIndex: 3 }}
      viewBox="-600 0 1200 400"
      preserveAspectRatio="xMidYMax meet"
    >
      <defs>
        <path id="leftPath" d={LEFT_PATH} />
        <path id="rightPath" d={RIGHT_PATH} />
      </defs>
      <path 
        d={LEFT_PATH} 
        stroke="white" 
        strokeWidth="1"
        strokeOpacity="0"
        fill="none" 
      />
      <path 
        d={RIGHT_PATH} 
        stroke="white" 
        strokeWidth="1"
        strokeOpacity="0"
        fill="none" 
      />
    </svg>
  );

  return (
    <div className="fixed inset-0 bg-black">
      <div className="absolute top-4 left-4 z-10 flex gap-2">
        {videoDevices.length > 1 && (
          <select
            className="bg-black/50 text-white px-4 py-2 rounded-full"
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
          >
            {videoDevices.map(device => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label}
              </option>
            ))}
          </select>
        )}
        <select
          className="bg-black/50 text-white px-4 py-2 rounded-full"
          value={rotation}
          onChange={(e) => setRotation(Number(e.target.value) as Rotation)}
        >
          <option value={0}>0°</option>
          <option value={90}>90°</option>
          <option value={180}>180°</option>
          <option value={270}>270°</option>
        </select>
      </div>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className={`absolute inset-0 w-full h-full object-cover ${
          frameQueueRef.current.length > 0 ? 'hidden' : ''
        }`}
        style={{ 
          transform: applyRotationStyle(rotation),
        }}
      />
      <canvas
        ref={croppedCanvasRef}
        width={FRAME_WIDTH}
        height={FRAME_HEIGHT}
        className="hidden"
      />
      <canvas
        ref={nextCanvasRef}
        width={FRAME_WIDTH}
        height={FRAME_HEIGHT}
        className="absolute inset-0 w-screen h-screen"
        style={{ 
          zIndex: 1,
          transform: applyRotationStyle(rotation),
          objectFit: 'cover'
        }}
      />
      <canvas
        ref={currentCanvasRef}
        width={FRAME_WIDTH}
        height={FRAME_HEIGHT}
        className="absolute inset-0 w-screen h-screen"
        style={{ 
          zIndex: 2,
          transform: applyRotationStyle(rotation),
          objectFit: 'cover'
        }}
      />

      <SvgPaths />

      <div className="fixed inset-0 overflow-visible pointer-events-none" style={{ zIndex: 3 }}>
        <svg 
          className="w-screen h-screen"
          viewBox="-600 0 1200 400"
          preserveAspectRatio="xMidYMax meet"
        >
          {transcripts.map((transcript) => {
            const age = (Date.now() - transcript.timestamp) / 1000;
            const progress = age / SCROLL_DURATION;
            const isLeft = transcript.timestamp % 2 === 0;
            
            return (
              <text
                key={transcript.timestamp}
                className="text-4xl font-bold  fill-white drop-shadow-lg"
                style={{
                  opacity: Math.max(0, 1 - progress * 1.5),
                  transition: 'opacity 16ms linear',
                  fontFamily: '"Sigmar", serif'
                }}
              >
                <textPath
                  href={isLeft ? "#leftPath" : "#rightPath"}
                  startOffset={isLeft ? `${progress * 100}%` : `${(1 - progress) * 100}%`}
                  textAnchor="middle"
                  className="fill-white"
                >
                  {transcript.text}
                </textPath>
              </text>
            );
          })}
        </svg>
      </div>

      <div 
        className="absolute left-1/2 bottom-16 -translate-x-1/2 z-20"
        style={{
          width: COIN_SIZE,
          height: COIN_SIZE,
          perspective: '1000px',
          transformStyle: 'preserve-3d',
        }}
      >
        <div
          style={{
            width: '100%',
            height: '100%',
            position: 'relative',
            transformStyle: 'preserve-3d',
            animation: `coin-rotate ${ROTATION_DURATION}s linear infinite`,
          }}
        >
          <img
            src="./mischief.jpg"
            alt="Mischief"
            className="absolute w-full h-full rounded-full object-cover"
            style={{
              backfaceVisibility: 'hidden',
              transform: 'rotateY(0deg)',
              filter: 'brightness(1.2)',
            }}
          />
          <img
            src="./mischief.jpg"
            alt="Mischief"
            className="absolute w-full h-full rounded-full object-cover"
            style={{
              backfaceVisibility: 'hidden',
              transform: 'rotateY(180deg)',
              filter: 'brightness(0.8)',
            }}
          />
        </div>
      </div>

      {wsStatus !== 'connected' && (
        <div className="absolute top-4 right-4 z-10 px-4 py-2 rounded-full bg-black/50 text-white">
          {wsStatus === 'connecting' ? 'Connecting...' : 'Reconnecting...'}
        </div>
      )}
    </div>
  );
};

export default WarpPage;
