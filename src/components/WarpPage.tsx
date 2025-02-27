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

  // Initialize webcam
  useEffect(() => {
    const initializeWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: {
            width: { ideal: FRAME_WIDTH },
            height: { ideal: FRAME_HEIGHT }
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
  }, []);

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

  return (
    <div className="fixed inset-0 bg-black">
      {wsStatus !== 'connected' && (
        <div className="absolute top-4 right-4 z-10 px-4 py-2 rounded-full bg-black/50 text-white">
          {wsStatus === 'connecting' ? 'Connecting...' : 'Reconnecting...'}
        </div>
      )}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className={`absolute inset-0 w-full h-full object-cover ${
          frameQueueRef.current.length > 0 ? 'hidden' : ''
        }`}
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
        className="absolute inset-0 w-full h-full object-cover"
        style={{ zIndex: 1 }}
      />
      <canvas
        ref={currentCanvasRef}
        width={FRAME_WIDTH}
        height={FRAME_HEIGHT}
        className="absolute inset-0 w-full h-full object-cover"
        style={{ zIndex: 2 }}
      />
    </div>
  );
};

export default WarpPage;
