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
const COIN_RELATIVE_SIZE = 0.09; // 9% of container width (increased for square layout)
const ROTATION_DURATION = 3; // Seconds for one full rotation
const PATH_BASE_Y = 300;  // Base Y position for the paths (adjusted for square layout)
const PATH_AMPLITUDE = 60;  // How much the path waves up and down
const DVD_DEBUG_MODE = false; // Set to true to have DVD mode always on (for testing)
const LEFT_PATH = `M 0 ${PATH_BASE_Y} C 100 ${PATH_BASE_Y + PATH_AMPLITUDE} 200 ${PATH_BASE_Y - PATH_AMPLITUDE} 300 ${PATH_BASE_Y} C 350 ${PATH_BASE_Y + PATH_AMPLITUDE} 400 ${PATH_BASE_Y - PATH_AMPLITUDE} 450 ${PATH_BASE_Y}`;
const RIGHT_PATH = `M 450 ${PATH_BASE_Y} C 500 ${PATH_BASE_Y - PATH_AMPLITUDE} 550 ${PATH_BASE_Y + PATH_AMPLITUDE} 600 ${PATH_BASE_Y} C 650 ${PATH_BASE_Y + PATH_AMPLITUDE} 700 ${PATH_BASE_Y - PATH_AMPLITUDE} 900 ${PATH_BASE_Y}`;

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

const PageContainer: React.FC<{ children: React.ReactNode; circles: Array<{id: number; x: number; y: number; size: number; color: string; vx: number; vy: number; sizeVel: number}> }> = ({ children, circles }) => {
  return (
    <div className="fixed inset-0 flex items-center justify-center overflow-hidden">
      {/* Full-screen animated background */}
      <div 
        className="absolute inset-0 w-full h-full"
        style={{
          zIndex: 0,
          background: 'linear-gradient(135deg, #1a8cd8 0%, #22A1EC 25%, #FFBAF1 75%, #e89dd9 100%)',
        }}
      />
      
      {/* Animated morphing circles - full screen */}
      <div className="absolute inset-0 w-full h-full" style={{ zIndex: 0 }}>
        {circles.map(circle => (
          <div
            key={circle.id}
            className="absolute rounded-full"
            style={{
              left: `${circle.x}%`,
              top: `${circle.y}%`,
              width: `${circle.size}vh`,
              height: `${circle.size}vh`,
              backgroundColor: circle.color,
              opacity: 0.7,
              transform: 'translate(-50%, -50%)',
              transition: 'all 0.05s linear',
              filter: 'blur(80px)',
              mixBlendMode: 'normal',
            }}
          />
        ))}
      </div>
      
      {/* Square content container */}
      <div 
        className="relative overflow-hidden"
        style={{
          width: 'min(100vh, 100vw)',
          height: 'min(100vh, 100vw)',
          aspectRatio: '1 / 1',
          zIndex: 1,
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
  const lastUpdateTime = useRef(Date.now());

  const videoRef = useRef<HTMLVideoElement>(null);
  const croppedCanvasRef = useRef<HTMLCanvasElement>(null);
  const currentCanvasRef = useRef<HTMLCanvasElement>(null);
  const nextCanvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const frameQueueRef = useRef<HTMLImageElement[]>([]);
  const isTransitioningRef = useRef(false);
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
  
  // Configuration panel state
  const [showConfig, setShowConfig] = useState(false);
  const [strength, setStrength] = useState(() => {
    const saved = localStorage.getItem('magicmirror_strength');
    return saved ? parseFloat(saved) : 0.375;
  });
  const [inferenceSteps, setInferenceSteps] = useState(() => {
    const saved = localStorage.getItem('magicmirror_steps');
    return saved ? parseInt(saved) : 8;
  });
  const [jpegQuality, setJpegQuality] = useState(() => {
    const saved = localStorage.getItem('magicmirror_quality');
    return saved ? parseInt(saved) : 90;
  });
  const [modelType, setModelType] = useState<'turbo' | 'lightning'>(() => {
    const saved = localStorage.getItem('magicmirror_model');
    return (saved as 'turbo' | 'lightning') || 'lightning';
  });
  const [isModelLoading, setIsModelLoading] = useState(false);
  
  // Animated background circles state
  const [circles, setCircles] = useState(() => 
    Array.from({ length: 10 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 30 + 20, // 20-50% of screen
      color: i % 2 === 0 ? '#22A1EC' : '#FFBAF1',
      vx: (Math.random() - 0.5) * 0.05,
      vy: (Math.random() - 0.5) * 0.05,
      sizeVel: (Math.random() - 0.5) * 0.1,
    }))
  );
  
  // DVD bouncing logo state
  const [isDvdMode, setIsDvdMode] = useState(false);
  const [logoPos, setLogoPos] = useState({ x: 50, y: 50 });
  const [logoVelocity, setLogoVelocity] = useState({ x: 0.045, y: 0.03 });
  const [logoColor, setLogoColor] = useState('#4A90E2');
  const [cornerHitEffect, setCornerHitEffect] = useState(false);
  const dvdModeStartTime = useRef<number>(0);
  const nextDvdModeTime = useRef<number>(Date.now() + 20 * 60 * 1000); // Start in 20 minutes

  // Animated background circles
  useEffect(() => {
    const animateCircles = setInterval(() => {
      setCircles(prevCircles => 
        prevCircles.map(circle => {
          let newX = circle.x + circle.vx;
          let newY = circle.y + circle.vy;
          let newSize = circle.size + circle.sizeVel;
          let newVx = circle.vx;
          let newVy = circle.vy;
          let newSizeVel = circle.sizeVel;
          
          // Bounce off edges
          if (newX < -20 || newX > 120) {
            newVx = -newVx;
            newX = newX < -20 ? -20 : 120;
          }
          if (newY < -20 || newY > 120) {
            newVy = -newVy;
            newY = newY < -20 ? -20 : 120;
          }
          
          // Oscillate size between 20-60%
          if (newSize < 20 || newSize > 60) {
            newSizeVel = -newSizeVel;
            newSize = newSize < 20 ? 20 : 60;
          }
          
          return {
            ...circle,
            x: newX,
            y: newY,
            size: newSize,
            vx: newVx,
            vy: newVy,
            sizeVel: newSizeVel,
          };
        })
      );
    }, 50); // Update every 50ms for smooth animation
    
    return () => clearInterval(animateCircles);
  }, []);

  // DVD Mode Logic
  useEffect(() => {
    // If debug mode is on, activate DVD mode immediately and keep it on
    if (DVD_DEBUG_MODE && !isDvdMode) {
      setIsDvdMode(true);
      dvdModeStartTime.current = Date.now();
      setLogoPos({ 
        x: Math.random() * 80 + 10, 
        y: Math.random() * 80 + 10 
      });
      setLogoVelocity({ 
        x: (Math.random() > 0.5 ? 1 : -1) * (Math.random() * 0.03 + 0.03), 
        y: (Math.random() > 0.5 ? 1 : -1) * (Math.random() * 0.03 + 0.03)
      });
      return;
    }
    
    // Normal timing mode (only if debug is off)
    if (DVD_DEBUG_MODE) return;
    
    const checkDvdMode = setInterval(() => {
      const now = Date.now();
      
      if (isDvdMode) {
        // Check if 5 minutes have passed
        if (now - dvdModeStartTime.current >= 5 * 60 * 1000) {
          setIsDvdMode(false);
          nextDvdModeTime.current = now + 20 * 60 * 1000; // Next DVD mode in 20 minutes
        }
      } else {
        // Check if it's time to start DVD mode
        if (now >= nextDvdModeTime.current) {
          setIsDvdMode(true);
          dvdModeStartTime.current = now;
          // Reset logo to random position with random velocity
          setLogoPos({ 
            x: Math.random() * 80 + 10, 
            y: Math.random() * 80 + 10 
          });
          setLogoVelocity({ 
            x: (Math.random() > 0.5 ? 1 : -1) * (Math.random() * 1.5 + 1), 
            y: (Math.random() > 0.5 ? 1 : -1) * (Math.random() * 1.5 + 1)
          });
        }
      }
    }, 1000); // Check every second
    
    return () => clearInterval(checkDvdMode);
  }, [isDvdMode]);

  // DVD Logo Animation
  useEffect(() => {
    if (!isDvdMode) return;
    
    const animate = () => {
      setLogoPos(prev => {
        const logoSizeX = 9; // Percentage of screen width (matches COIN_RELATIVE_SIZE * 100)
        const logoSizeY = 9; // Percentage of screen height
        
        // Check if we're in the last 30 seconds - force corner hit
        const timeInDvdMode = Date.now() - dvdModeStartTime.current;
        const timeRemaining = (5 * 60 * 1000) - timeInDvdMode;
        const forceCorner = !DVD_DEBUG_MODE && timeRemaining < 30000; // Last 30 seconds
        
        let newX = prev.x;
        let newY = prev.y;
        let newVelX = logoVelocity.x;
        let newVelY = logoVelocity.y;
        
        if (forceCorner) {
          // Find nearest corner and drift toward it
          const corners = [
            { x: 0, y: 0 },
            { x: 100 - logoSizeX, y: 0 },
            { x: 0, y: 100 - logoSizeY },
            { x: 100 - logoSizeX, y: 100 - logoSizeY }
          ];
          
          // Find closest corner
          let closestCorner = corners[0];
          let minDist = Infinity;
          corners.forEach(corner => {
            const dist = Math.sqrt(Math.pow(corner.x - prev.x, 2) + Math.pow(corner.y - prev.y, 2));
            if (dist < minDist) {
              minDist = dist;
              closestCorner = corner;
            }
          });
          
          // Move toward corner with acceleration
          const speed = 0.3; // Faster movement toward corner
          newX = prev.x + (closestCorner.x - prev.x) * speed;
          newY = prev.y + (closestCorner.y - prev.y) * speed;
          
          // Trigger effect if we're close enough to corner
          if (Math.abs(newX - closestCorner.x) < 1 && Math.abs(newY - closestCorner.y) < 1) {
            newX = closestCorner.x;
            newY = closestCorner.y;
            setCornerHitEffect(true);
            setTimeout(() => setCornerHitEffect(false), 3000);
          }
        } else {
          // Normal bouncing physics
          newX = prev.x + logoVelocity.x;
          newY = prev.y + logoVelocity.y;
          let bounced = false;
          
          // Check bounds and bounce
          if (newX <= 0 || newX >= (100 - logoSizeX)) {
            newVelX = -newVelX;
            newX = newX <= 0 ? 0 : (100 - logoSizeX);
            bounced = true;
          }
          
          if (newY <= 0 || newY >= (100 - logoSizeY)) {
            newVelY = -newVelY;
            newY = newY <= 0 ? 0 : (100 - logoSizeY);
            bounced = true;
          }
          
          // Check for corner hit (both X and Y bounced at the same time)
          if (bounced && newVelX !== logoVelocity.x && newVelY !== logoVelocity.y) {
            setCornerHitEffect(true);
            setTimeout(() => setCornerHitEffect(false), 3000);
          }
          
          // Change color on bounce
          if (bounced) {
            const colors = ['#4A90E2', '#E24A90', '#90E24A', '#E2904A', '#904AE2', '#4AE290'];
            setLogoColor(colors[Math.floor(Math.random() * colors.length)]);
          }
          
          if (newVelX !== logoVelocity.x || newVelY !== logoVelocity.y) {
            setLogoVelocity({ x: newVelX, y: newVelY });
          }
        }
        
        return { x: newX, y: newY };
      });
    };
    
    const interval = setInterval(animate, 16); // ~60fps
    return () => clearInterval(interval);
  }, [isDvdMode, logoVelocity]);

  // Keyboard handler for config panel
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.key === 'c' || e.key === 'C') {
        setShowConfig(prev => !prev);
      }
    };
    
    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, []);

  // Update backend when settings change
  const updateBackendSetting = useCallback(async (endpoint: string, value: number | string) => {
    if (!serverPassword) return;
    
    try {
      const response = await fetch(`${buildPromptEndpointUrl().replace('/prompt/', '')}/${endpoint}/${value}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${serverPassword}`,
        },
      });
      
      if (response.ok) {
        console.log(`Updated ${endpoint} to ${value}`);
      }
    } catch (error) {
      console.error(`Failed to update ${endpoint}:`, error);
    }
  }, [serverPassword]);

  // Save and sync settings
  useEffect(() => {
    localStorage.setItem('magicmirror_strength', strength.toString());
    updateBackendSetting('strength', strength);
  }, [strength, updateBackendSetting]);

  useEffect(() => {
    localStorage.setItem('magicmirror_steps', inferenceSteps.toString());
    updateBackendSetting('steps', inferenceSteps);
  }, [inferenceSteps, updateBackendSetting]);

  useEffect(() => {
    localStorage.setItem('magicmirror_quality', jpegQuality.toString());
    updateBackendSetting('jpeg_quality', jpegQuality);
  }, [jpegQuality, updateBackendSetting]);

  useEffect(() => {
    localStorage.setItem('magicmirror_model', modelType);
    
    // Trigger model reload on backend
    const reloadModel = async () => {
      if (!serverPassword) return;
      
      setIsModelLoading(true);
      try {
        const response = await fetch(`${buildPromptEndpointUrl().replace('/prompt/', '')}/reload_model/${modelType}`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${serverPassword}`,
          },
        });
        
        const result = await response.json();
        console.log(`Model reload response:`, result);
        
        if (result.status === 'success') {
          console.log(`✅ Model reloaded to ${modelType}`);
        }
      } catch (error) {
        console.error(`Failed to reload model:`, error);
      } finally {
        // Wait a moment before clearing loading state
        setTimeout(() => setIsModelLoading(false), 2000);
      }
    };
    
    reloadModel();
  }, [modelType, serverPassword]);

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
      const timeSinceLastTransition = now - lastUpdateTime.current;

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
              
              lastUpdateTime.current = Date.now();
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

  // WebRTC Connection
  const connectWebRTC = useCallback(async () => {
    if (!serverPassword) return;
    
    try {
      setRtcStatus('connecting');
      console.log('🎥 Starting WebRTC connection...');
      
      // Create peer connection
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
      });
      pcRef.current = pc;
      
      // Handle incoming video track
      pc.ontrack = (event) => {
        console.log('📺 Received video track', event);
        console.log('Stream:', event.streams[0]);
        console.log('Track kind:', event.track.kind);
        
        const stream = event.streams[0];
        
        if (webrtcVideoRef.current) {
          console.log('Setting video srcObject');
          webrtcVideoRef.current.srcObject = stream;
          webrtcVideoRef.current.onloadedmetadata = () => {
            console.log('✅ Video metadata loaded');
            console.log('Video dimensions:', webrtcVideoRef.current?.videoWidth, 'x', webrtcVideoRef.current?.videoHeight);
          };
          webrtcVideoRef.current.onplay = () => {
            console.log('✅ Video is playing');
            setRtcStatus('connected');
          };
          webrtcVideoRef.current.play()
            .then(() => console.log('✅ Video play() succeeded'))
            .catch(e => console.error('❌ Error playing video:', e));
        } else {
          console.error('❌ webrtcVideoRef.current is null!');
        }
      };
      
      pc.onconnectionstatechange = () => {
        console.log(`WebRTC connection state: ${pc.connectionState}`);
        setRtcStatus(pc.connectionState as any);
        if (pc.connectionState === 'failed') {
          console.error('❌ WebRTC connection failed!');
          // Don't fallback - we want to debug WebRTC
        }
      };
      
      pc.oniceconnectionstatechange = () => {
        console.log(`ICE connection state: ${pc.iceConnectionState}`);
      };
      
      pc.onicegatheringstatechange = () => {
        console.log(`ICE gathering state: ${pc.iceGatheringState}`);
      };
      
      // Add webcam video track to send to server
      const videoTrack = currentStream?.getVideoTracks()[0];
      if (videoTrack) {
        pc.addTrack(videoTrack, currentStream);
        console.log('📤 Added webcam track to peer connection');
      }
      
      // Add transceiver to receive AI video from server
      pc.addTransceiver('video', { direction: 'recvonly' });
      
      // Create offer
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      
      console.log('Offer created and set');
      
      // Send offer to server
      const response = await fetch(`${buildPromptEndpointUrl().replace('/prompt/', '')}/webrtc/offer`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${serverPassword}`,
        },
        body: JSON.stringify({
          sdp: pc.localDescription?.sdp,
          type: pc.localDescription?.type,
        }),
      });
      
      if (!response.ok) {
        throw new Error(`WebRTC offer failed: ${response.statusText}`);
      }
      
      const answer = await response.json();
      console.log('📥 Received answer from server:', answer);
      
      if (!answer.sdp || !answer.type) {
        throw new Error('Invalid answer from server: ' + JSON.stringify(answer));
      }
      
      // Set remote description
      await pc.setRemoteDescription(new RTCSessionDescription(answer));
      
      console.log('✅ WebRTC signaling complete, waiting for connection...');
      
    } catch (error) {
      console.error('❌ WebRTC setup error:', error);
      if (error instanceof Error) {
        console.error('Error message:', error.message);
        console.error('Error stack:', error.stack);
      }
      setRtcStatus('failed');
    }
  }, [serverPassword]);

  const connectWebSocket = useCallback(async () => {
    if (!serverPassword) {
      return;
    }
    console.log('📡 Connecting to WebSocket...');
    const ws = new WebSocket(buildWebsocketUrl());
    
    ws.onopen = () => {
      console.log("WebSocket connected");
      // Send authentication immediately after connection
      ws.send(JSON.stringify({
        type: "auth",
        password: serverPassword
      }));
    };

    ws.onmessage = (event) => {
      if (event.data instanceof Blob) {
        // Update last update time when receiving a frame
        lastUpdateTime.current = Date.now();
        // Handle binary frame data
        const blob = event.data;
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {
          URL.revokeObjectURL(url);
          frameQueueRef.current.push(img);
        };
        img.src = url;
      }
    };

    ws.onclose = () => {
      console.log("WebSocket closed");
      setWsStatus('disconnected');
      // Try to reconnect
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
      reconnectTimeoutRef.current = setTimeout(connectWebSocket, 1000);
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };

    socketRef.current = ws;
    setWsStatus('connected');
  }, [serverPassword]);

  // Add new useEffect to check for stale updates
  useEffect(() => {
    const checkStaleUpdates = () => {
      const now = Date.now();
      const timeSinceLastUpdate = now - lastUpdateTime.current;
      const FIVE_MINUTES = 5 * 60 * 1000; // 5 minutes in milliseconds

      if (timeSinceLastUpdate > FIVE_MINUTES) {
        console.log("No updates received for 5 minutes, refreshing page...");
        window.location.reload();
      }
    };

    // Check every minute
    const staleCheckInterval = setInterval(checkStaleUpdates, 60 * 1000);

    return () => {
      clearInterval(staleCheckInterval);
    };
  }, []);

  // Modify transcribe function
  const transcribe = async (audioBlob: Blob) => {
    if (!serverPassword) {
      return;
    }

    const formData = new FormData();
    formData.append('file', audioBlob);

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

  // Send frames via WebSocket
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
    const targetAspect = 1; // Force 1:1 square aspect ratio

    // Use the smallest dimension to ensure it fits
    const size = Math.min(windowWidth, windowHeight);

    return { width: size, height: size };
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

  // Modify the return statement to show both video and auth overlay
  return (
    <PageContainer circles={circles}>
      {/* Webcam preview video */}
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

      {/* Canvas elements for rendering */}
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

          <div className="absolute bottom-0 left-0 right-0 overflow-hidden" style={{ zIndex: 3, height: '40%' }}>
            <svg 
              className="w-full h-full"
              viewBox="0 0 900 400"
              preserveAspectRatio="xMidYMid meet"
            >
              {transcripts.map((transcript) => {
                const age = (Date.now() - transcript.timestamp) / 1000;
                const progress = age / SCROLL_DURATION;
                const isLeft = transcript.timestamp % 2 === 0;
                
                const startOffset = isLeft ? progress * 100 : (1 - progress) * 100;
                
                return (
                  <text
                    key={transcript.timestamp}
                    className="text-3xl font-bold fill-white drop-shadow-lg"
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
            className="absolute z-20"
            style={{
              width: `${calculateCanvasDimensions().width * COIN_RELATIVE_SIZE}px`,
              height: `${calculateCanvasDimensions().width * COIN_RELATIVE_SIZE}px`,
              left: isDvdMode ? `${logoPos.x}%` : '50%',
              bottom: isDvdMode ? 'auto' : '4%',
              top: isDvdMode ? `${logoPos.y}%` : 'auto',
              transform: isDvdMode ? 'none' : 'translateX(-50%)',
              perspective: '1000px',
              transformStyle: 'preserve-3d',
              transition: isDvdMode ? 'none' : 'all 0.5s ease-out',
              filter: cornerHitEffect ? 'drop-shadow(0 0 30px currentColor) drop-shadow(0 0 60px currentColor) brightness(1.5)' : 'none',
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
                animation: `coin-rotate ${cornerHitEffect ? ROTATION_DURATION / 2 : ROTATION_DURATION}s linear infinite`,
                color: isDvdMode ? logoColor : undefined,
              }}
            >
              <div className="absolute inset-0" style={{ backfaceVisibility: 'hidden', transform: 'rotateY(0deg)' }}>
                <img
                  src="./mischief.svg"
                  alt="Mischief"
                  className="absolute w-full h-full"
                  style={{ filter: isDvdMode ? 'none' : 'url(#blue-tint)' }}
                />
              </div>
              <div className="absolute inset-0" style={{ backfaceVisibility: 'hidden', transform: 'rotateY(180deg)' }}>
                <img
                  src="./mischief.svg"
                  alt="Mischief"
                  className="absolute w-full h-full"
                  style={{ filter: isDvdMode ? 'none' : 'url(#pink-tint)' }}
                />
              </div>
            </div>
          </div>
          
          {/* Corner Hit Effect - Radial glow */}
          {cornerHitEffect && (
            <div
              className="absolute inset-0 pointer-events-none z-10"
              style={{
                background: `radial-gradient(circle at ${logoPos.x}% ${logoPos.y}%, ${logoColor}80 0%, ${logoColor}40 10%, transparent 30%)`,
                animation: 'pulse 0.5s ease-out 6',
              }}
            />
          )}

          {/* Configuration Panel */}
          {showConfig && (
            <div className="absolute top-4 right-4 z-50 bg-black/90 backdrop-blur-md p-6 rounded-lg shadow-2xl border border-white/20" style={{ minWidth: '300px' }}>
              <div className="space-y-4">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-white font-bold text-lg">Settings</h3>
                  <button 
                    onClick={() => setShowConfig(false)}
                    className="text-white/60 hover:text-white"
                  >
                    ✕
                  </button>
                </div>
                
                {/* Strength */}
                <div>
                  <label className="text-white/80 text-sm block mb-2">
                    Strength: {(strength * 100).toFixed(1)}%
                  </label>
                  <input
                    type="range"
                    min="0.125"
                    max="0.875"
                    step="0.125"
                    value={strength}
                    onChange={(e) => setStrength(parseFloat(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-white/50 text-xs mt-1">
                    Lightning optimal: 12.5%, 25%, 37.5%, 50%, 62.5%, 75%, 87.5%
                  </div>
                </div>

                {/* Inference Steps */}
                <div>
                  <label className="text-white/80 text-sm block mb-2">
                    Inference Steps: {inferenceSteps}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="16"
                    step="1"
                    value={inferenceSteps}
                    onChange={(e) => setInferenceSteps(parseInt(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-white/50 text-xs mt-1">
                    Lower = faster, Higher = better quality
                  </div>
                </div>

                {/* JPEG Quality */}
                <div>
                  <label className="text-white/80 text-sm block mb-2">
                    JPEG Quality: {jpegQuality}
                  </label>
                  <input
                    type="range"
                    min="50"
                    max="100"
                    step="5"
                    value={jpegQuality}
                    onChange={(e) => setJpegQuality(parseInt(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-white/50 text-xs mt-1">
                    Higher = better quality, larger files
                  </div>
                </div>

                {/* Model Type */}
                <div>
                  <label className="text-white/80 text-sm block mb-2">
                    Model Type
                  </label>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setModelType('lightning')}
                      disabled={isModelLoading}
                      className={`flex-1 px-3 py-2 rounded text-sm ${
                        modelType === 'lightning'
                          ? 'bg-blue-500 text-white'
                          : 'bg-white/10 text-white/60 hover:bg-white/20'
                      } ${isModelLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      Lightning
                    </button>
                    <button
                      onClick={() => setModelType('turbo')}
                      disabled={isModelLoading}
                      className={`flex-1 px-3 py-2 rounded text-sm ${
                        modelType === 'turbo'
                          ? 'bg-blue-500 text-white'
                          : 'bg-white/10 text-white/60 hover:bg-white/20'
                      } ${isModelLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                      Turbo
                    </button>
                  </div>
                  <div className="text-white/50 text-xs mt-1">
                    {isModelLoading ? '🔄 Reloading model...' : '✓ Hot-swappable'}
                  </div>
                </div>

                <div className="text-white/40 text-xs pt-2 border-t border-white/10">
                  Press 'C' to toggle this panel
                </div>
              </div>
            </div>
          )}

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
