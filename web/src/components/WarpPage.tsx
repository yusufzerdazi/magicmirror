import React, { useState, useEffect, useRef, useCallback } from 'react';
import PasswordAuth from './PasswordAuth';

const FRAME_WIDTH = 512;
const FRAME_HEIGHT = 512;
const INITIAL_PROMPT = "a mischievous cat with a third eye, matte pastel colour pallete in a cartoon style";
const FRAME_INTERVAL = 250; // Send 4 frames per second
const MAX_BUFFER_SIZE = 16; // Don't let buffer grow too large
const DISPLAY_DURATION = 0; // Time to show each frame before transition
const AUDIO_INTERVAL = 10000; // Process audio every 5 seconds
const SCROLL_DURATION = 5; // Seconds to keep text visible
const COIN_RELATIVE_SIZE = 0.05; // 10% of container width
const ROTATION_DURATION = 3; // Seconds for one full rotation
const PATH_BASE_Y = 350;  // Base Y position for the paths
const PATH_AMPLITUDE = 50;  // How much the path waves up and down
const LEFT_PATH = `M 0 ${PATH_BASE_Y} C 100 ${PATH_BASE_Y + PATH_AMPLITUDE} 200 ${PATH_BASE_Y - PATH_AMPLITUDE} 300 ${PATH_BASE_Y} C 350 ${PATH_BASE_Y + PATH_AMPLITUDE} 400 ${PATH_BASE_Y - PATH_AMPLITUDE} 450 ${PATH_BASE_Y}`;
const RIGHT_PATH = `M 450 ${PATH_BASE_Y} C 500 ${PATH_BASE_Y - PATH_AMPLITUDE} 550 ${PATH_BASE_Y + PATH_AMPLITUDE} 600 ${PATH_BASE_Y} C 650 ${PATH_BASE_Y + PATH_AMPLITUDE} 700 ${PATH_BASE_Y - PATH_AMPLITUDE} 900 ${PATH_BASE_Y}`;

// Add new constants for reconnection and health check
const MAX_RECONNECT_ATTEMPTS = 10;
const INITIAL_RECONNECT_DELAY = 1000; // 1 second
const MAX_RECONNECT_DELAY = 30000; // 30 seconds
const HEALTH_CHECK_INTERVAL = 30000; // 30 seconds
const MAX_NO_UPDATES_TIME = 300000; // 5 minutes
const PING_INTERVAL = 10000; // 10 seconds

// Add new constant for env password
const ENV_PASSWORD = import.meta.env.VITE_WARP_PASSWORD;

type TranscriptEntry = {
  text: string;
  timestamp: number;
};

type DeviceInfo = {
  deviceId: string;
  label: string;
};

type Rotation = 0 | 90 | 180 | 270;

const buildWebsocketUrl = () => {
  return `${import.meta.env.VITE_API_URL}:8765`;
};

const buildPromptEndpointUrl = () => {
  return `${import.meta.env.VITE_API_URL}:5556/prompt/`;
};

// First, move the PasswordAuth component to be an overlay
const PasswordOverlay: React.FC<{ onAuthenticated: (password: string) => void }> = ({ onAuthenticated }) => {
  return (
    <div className="fixed inset-0 flex items-center justify-center" style={{ zIndex: 100 }}>
      <div className="absolute inset-0 bg-black/50" />
      <div className="relative z-10">
        <PasswordAuth onAuthenticated={onAuthenticated} />
      </div>
    </div>
  );
};

const PageContainer: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return (
    <div className="fixed inset-0 bg-black flex items-center justify-center">
      <div 
        className="relative bg-black h-screen"
        style={{
          width: 'calc(16/9 * 100vh)', // Force 9:16 ratio based on height
          maxWidth: '100vw',           // Don't overflow viewport width
        }}
      >
        {children}
      </div>
    </div>
  );
};

const WarpPage = () => {
  const [currentStream, setCurrentStream] = useState<MediaStream | null>(null);
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
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const [transcripts, setTranscripts] = useState<TranscriptEntry[]>([]);
  const [videoDevices, setVideoDevices] = useState<DeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [rotation, setRotation] = useState<Rotation>(0);
  const [totalTranscripts, setTotalTranscripts] = useState<number>(0);
  const [isFirstCanvasCurrent, setIsFirstCanvasCurrent] = useState(true);
  const [serverPassword, setServerPassword] = useState<string | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [authToken, setAuthToken] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const reconnectAttemptsRef = useRef(0);
  const lastFrameTimeRef = useRef(Date.now());
  const healthCheckIntervalRef = useRef<NodeJS.Timeout>();
  const pingIntervalRef = useRef<NodeJS.Timeout>();
  const noUpdatesTimeoutRef = useRef<NodeJS.Timeout>();

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

  // Move video initialization logic outside of authentication check
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
            deviceId: { exact: selectedDeviceId }
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

  // Send initial prompt when warp is ready and authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      return;
    }

    const promptEndpointUrl = buildPromptEndpointUrl();
    const encodedPrompt = encodeURIComponent(INITIAL_PROMPT);
    const endpoint = `${promptEndpointUrl}${encodedPrompt}`;

    fetch(endpoint, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${serverPassword}`
      }
    }).catch(error => {
      console.error('Error sending initial prompt:', error);
    });
  }, [isAuthenticated, serverPassword]);

  // Frame display logic
  useEffect(() => {
    if (!isRendering) return;

    const displayNextFrame = () => {
      const now = Date.now();
      const timeSinceLastTransition = now - lastTransitionTime.current;

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
          // Clear the canvas
          nextCtx.clearRect(0, 0, nextCanvas.width, nextCanvas.height);
          
          // Calculate dimensions to cover the canvas while maintaining aspect ratio
          const frameAspect = nextFrame.width / nextFrame.height;
          const canvasAspect = nextCanvas.width / nextCanvas.height;
          
          let drawWidth = nextCanvas.width;
          let drawHeight = nextCanvas.height;
          let offsetX = 0;
          let offsetY = 0;

          if (canvasAspect > frameAspect) {
            // Canvas is wider than frame - match width and overflow height
            drawWidth = nextCanvas.width;
            drawHeight = drawWidth / frameAspect;
            offsetY = (nextCanvas.height - drawHeight) / 2;
          } else {
            // Canvas is taller than frame - match height and overflow width
            drawHeight = nextCanvas.height;
            drawWidth = drawHeight * frameAspect;
            offsetX = (nextCanvas.width - drawWidth) / 2;
          }

          // Draw the frame with cover behavior
          nextCtx.drawImage(
            nextFrame,
            offsetX, offsetY,
            drawWidth,
            drawHeight
          );
          
          // Start transition
          isTransitioningRef.current = true;
          let opacity = 1;
          
          const fade = () => {
            if (opacity > 0 && currentCanvasRef.current) {
              opacity -= 0.05;
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
  }, [isRendering, isFirstCanvasCurrent]);

  // Modify WebSocket connection logic
  const connectWebSocket = useCallback(async () => {
    if (!serverPassword) {
      return;
    }

    // Clear any existing timeouts
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    if (healthCheckIntervalRef.current) {
      clearInterval(healthCheckIntervalRef.current);
    }
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
    }
    if (noUpdatesTimeoutRef.current) {
      clearTimeout(noUpdatesTimeoutRef.current);
    }

    // Close existing socket if any
    if (socketRef.current) {
      socketRef.current.close();
    }

    const ws = new WebSocket(buildWebsocketUrl());
    
    ws.onopen = () => {
      console.log("WebSocket connected");
      // Reset reconnection attempts on successful connection
      reconnectAttemptsRef.current = 0;
      // Send authentication immediately after connection
      ws.send(JSON.stringify({
        type: "auth",
        password: serverPassword
      }));
      setWsStatus('connected');

      // Start health check
      healthCheckIntervalRef.current = setInterval(() => {
        const timeSinceLastFrame = Date.now() - lastFrameTimeRef.current;
        if (timeSinceLastFrame > HEALTH_CHECK_INTERVAL) {
          console.log("No frames received for too long, reconnecting...");
          ws.close();
        }
      }, HEALTH_CHECK_INTERVAL);

      // Start ping interval
      pingIntervalRef.current = setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: "ping" }));
        }
      }, PING_INTERVAL);

      // Reset no updates timeout
      noUpdatesTimeoutRef.current = setTimeout(() => {
        console.log("No updates for 5 minutes, refreshing page...");
        window.location.reload();
      }, MAX_NO_UPDATES_TIME);
    };

    ws.onmessage = (event) => {
      if (event.data instanceof Blob) {
        // Handle binary frame data
        const blob = event.data;
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          URL.revokeObjectURL(url);
          frameQueueRef.current.push(img);
          lastFrameTimeRef.current = Date.now();
        };
        img.src = url;
      } else {
        // Handle text messages (like pong)
        try {
          const data = JSON.parse(event.data);
          if (data.type === "pong") {
            lastFrameTimeRef.current = Date.now();
          }
        } catch (e) {
          console.error("Error parsing WebSocket message:", e);
        }
      }
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
      setWsStatus('disconnected');
      
      // Calculate next reconnect delay with exponential backoff
      const delay = Math.min(
        INITIAL_RECONNECT_DELAY * Math.pow(2, reconnectAttemptsRef.current),
        MAX_RECONNECT_DELAY
      );
      
      reconnectAttemptsRef.current++;
      
      if (reconnectAttemptsRef.current <= MAX_RECONNECT_ATTEMPTS) {
        console.log(`Attempting to reconnect in ${delay}ms (attempt ${reconnectAttemptsRef.current}/${MAX_RECONNECT_ATTEMPTS})`);
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, delay);
      } else {
        console.log("Max reconnection attempts reached, refreshing page...");
        window.location.reload();
      }
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    socketRef.current = ws;
  }, [serverPassword]);

  // Modify transcribe function
  const transcribe = async (audioBlob: Blob) => {
    if (!serverPassword) {
      return;
    }

    const formData = new FormData();
    formData.append('audio', audioBlob, 'audio.webm');

    try {
      const response = await fetch(`${import.meta.env.VITE_API_URL}:5556/transcribe`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${serverPassword}`
        },
        body: formData
      });

      if (response.status === 401) {
        setIsAuthenticated(false);
        setServerPassword(null);
        return;
      }

      const data = await response.json();
      if (data.text) {
        addTranscript(data.text);
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  // Update useEffect to depend on serverPassword
  useEffect(() => {
    if (!currentStream || !serverPassword) return;
    
    connectWebSocket();
    
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [currentStream, serverPassword, connectWebSocket]);

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

  // Update the audio initialization effect
  useEffect(() => {
    if (!isAuthenticated) return;

    const initAudio = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          }
        });

        // Remove Web Audio API setup
        // Just keep the MediaRecorder setup
        const mediaRecorder = new MediaRecorder(stream, {
          mimeType: MediaRecorder.isTypeSupported('audio/webm') ? 'audio/webm' : 'audio/ogg'
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

          // Play back the recorded audio
          const audioUrl = URL.createObjectURL(audioBlob);
          const audio = new Audio(audioUrl);
          audio.play();
          
          // Clean up the URL after playback
          audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
          };

          // Send for transcription
          await transcribe(audioBlob);
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
        console.error('Error accessing microphone:', error);
      }
    };

    initAudio();
  }, [isAuthenticated]);

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
    <div className="absolute inset-0" style={{ zIndex: 3 }}>
      <svg 
        className="w-full h-full pointer-events-none"
        viewBox="0 0 900 400"
        preserveAspectRatio="xMidYMax meet"
      >
        <defs>
          {/* Define gradients */}
          <linearGradient id="leftFade" x1="0%" y1="0%" x2="40%" y2="0%">
            <stop offset="0%" stopColor="white" stopOpacity="1" />
            <stop offset="100%" stopColor="white" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="rightFade" x1="100%" y1="0%" x2="60%" y2="0%">
            <stop offset="0%" stopColor="white" stopOpacity="1" />
            <stop offset="100%" stopColor="white" stopOpacity="0" />
          </linearGradient>

          {/* Define masks */}
          <mask id="leftMask">
            <rect x="0" y="0" width="900" height="400" fill="url(#leftFade)" />
          </mask>
          <mask id="rightMask">
            <rect x="0" y="0" width="900" height="400" fill="url(#rightFade)" />
          </mask>

          {/* Define paths */}
          <path id="leftPath" d={LEFT_PATH} />
          <path id="rightPath" d={RIGHT_PATH} />
        </defs>
      </svg>
    </div>
  );

  const firstCanvasStyle = {
    zIndex: isFirstCanvasCurrent ? 2 : 1,
    transform: applyRotationStyle(rotation),
    opacity: 1  // Remove transition, just keep opacity at 1
  };

  const secondCanvasStyle = {
    zIndex: isFirstCanvasCurrent ? 1 : 2,
    transform: applyRotationStyle(rotation),
    opacity: 1  // Remove transition, just keep opacity at 1
  };

  // Check for env password on mount
  useEffect(() => {
    if (ENV_PASSWORD) {
      handleAuthenticated(ENV_PASSWORD);
    }
  }, []);

  const handleAuthenticated = (password: string) => {
    setIsAuthenticated(true);
    setAuthToken(password);
    setServerPassword(password);
  };

  // Update the calculateCanvasDimensions function
  const calculateCanvasDimensions = () => {
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    const targetAspect = 16/9; // Force 9:16 aspect ratio

    let width, height;
    if (windowWidth / windowHeight > targetAspect) {
      // Window is wider than 9:16
      height = windowHeight;
      width = height * targetAspect;
    } else {
      // Window is taller than 9:16
      width = windowWidth;
      height = width / targetAspect;
    }

    return { width, height };
  };

  // Update the resize handler
  useEffect(() => {
    const updateDimensions = () => {
      const { width, height } = calculateCanvasDimensions();
      if (currentCanvasRef.current) {
        currentCanvasRef.current.width = width;
        currentCanvasRef.current.height = height;
      }
      if (nextCanvasRef.current) {
        nextCanvasRef.current.width = width;
        nextCanvasRef.current.height = height;
      }
    };

    window.addEventListener('resize', updateDimensions);
    updateDimensions(); // Set initial dimensions

    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Add keyboard listener for 'H' key
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === 'h') {
        setShowPreview(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, []);

  // Cleanup function
  useEffect(() => {
    return () => {
      // Cleanup all intervals and timeouts
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      if (healthCheckIntervalRef.current) {
        clearInterval(healthCheckIntervalRef.current);
      }
      if (pingIntervalRef.current) {
        clearInterval(pingIntervalRef.current);
      }
      if (noUpdatesTimeoutRef.current) {
        clearTimeout(noUpdatesTimeoutRef.current);
      }
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, []);

  // Modify the return statement to show both video and auth overlay
  return (
    <PageContainer>
      {/* Move video inside container */}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="absolute inset-0 w-full h-full object-cover"
        style={{ 
          transform: applyRotationStyle(rotation),
          visibility: frameQueueRef.current.length > 0 ? 'hidden' : 'visible',
          zIndex: 1
        }}
      />

      {/* Move canvases inside container */}
      <canvas
        ref={croppedCanvasRef}
        width={FRAME_WIDTH}
        height={FRAME_HEIGHT}
        className="hidden"
      />
      <canvas
        ref={nextCanvasRef}
        className="absolute inset-0 w-full h-full"
        style={secondCanvasStyle}
      />
      <canvas
        ref={currentCanvasRef}
        className="absolute inset-0 w-full h-full"
        style={firstCanvasStyle}
      />

      {/* Only show password overlay if no env password and not authenticated */}
      {!isAuthenticated && !ENV_PASSWORD ? (
        <PasswordOverlay onAuthenticated={handleAuthenticated} />
      ) : (
        <>
          {/* Rest of the authenticated UI */}
          {showPreview && (
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
          )}

          <SvgPaths />

          <div className="absolute inset-0 overflow-hidden" style={{ zIndex: 3 }}>
            <svg 
              className="w-full h-full"
              viewBox="0 0 900 400"
              preserveAspectRatio="xMidYMax meet"
            >
              {transcripts.map((transcript) => {
                const age = (Date.now() - transcript.timestamp) / 1000;
                const progress = age / SCROLL_DURATION;
                const isLeft = transcript.timestamp % 2 === 0;
                
                const startOffset = isLeft ? progress * 100 : (1 - progress) * 100;
                
                return (
                  <text
                    key={transcript.timestamp}
                    className="text-lg font-bold fill-white drop-shadow-lg"
                    style={{
                      fontFamily: '"Sigmar", serif',
                    }}
                    mask={isLeft ? "url(#leftMask)" : "url(#rightMask)"}
                  >
                    <textPath
                      href={isLeft ? "#leftPath" : "#rightPath"}
                      startOffset={`${startOffset}%`}
                      textAnchor="middle"
                    >
                      {transcript.text}
                    </textPath>
                  </text>
                );
              })}
            </svg>
          </div>

          <div 
            className="absolute left-1/2 -translate-x-1/2 z-20"
            style={{
              width: `${calculateCanvasDimensions().width * COIN_RELATIVE_SIZE}px`,
              height: `${calculateCanvasDimensions().width * COIN_RELATIVE_SIZE}px`,
              bottom: '4%',
              perspective: '1000px',
              transformStyle: 'preserve-3d',
            }}
          >
            <svg width="0" height="0">
              <defs>
                <filter id="blue-tint">
                  <feFlood floodColor="#22A1EC" result="tint" />
                  <feComposite in="tint" in2="SourceGraphic" operator="atop" />
                </filter>
                <filter id="pink-tint">
                  <feFlood floodColor="#FFBAF1" result="tint" />
                  <feComposite in="tint" in2="SourceGraphic" operator="atop" />
                </filter>
              </defs>
            </svg>
            <div
              style={{
                width: '100%',
                height: '100%',
                position: 'relative',
                transformStyle: 'preserve-3d',
                animation: `coin-rotate ${ROTATION_DURATION}s linear infinite`,
              }}
            >
              <div className="absolute inset-0" style={{ backfaceVisibility: 'hidden', transform: 'rotateY(0deg)' }}>
                <img
                  src="./mischief.svg"
                  alt="Mischief"
                  className="absolute w-full h-full"
                  style={{ filter: 'url(#blue-tint)' }}
                />
              </div>
              <div className="absolute inset-0" style={{ backfaceVisibility: 'hidden', transform: 'rotateY(180deg)' }}>
                <img
                  src="./mischief.svg"
                  alt="Mischief"
                  className="absolute w-full h-full"
                  style={{ filter: 'url(#pink-tint)' }}
                />
              </div>
            </div>
          </div>

          {showPreview && wsStatus !== 'connected' && (
            <div className="absolute top-4 right-4 z-10 px-4 py-2 rounded-full bg-black/50 text-white">
              {wsStatus === 'connecting' ? 'Connecting...' : 'Reconnecting...'}
            </div>
          )}

          {/* Add preview video */}
          {showPreview && frameQueueRef.current.length > 0 && (
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute top-4 right-4 rounded-lg shadow-lg z-50"
              style={{ 
                width: '200px',
                transform: applyRotationStyle(rotation),
                opacity: 0.8,
              }}
            />
          )}
        </>
      )}
    </PageContainer>
  );
};

export default WarpPage;
