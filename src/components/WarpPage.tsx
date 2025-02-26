import React, { useState, useEffect, useRef } from 'react';
import { createFullEndpoint } from '#root/utils/apiUtils.ts';
import useConditionalAuth from '#root/src/hooks/useConditionalAuth';
import { IS_WARP_LOCAL } from '#root/utils/constants.ts';

const FRAME_WIDTH = 512;
const FRAME_HEIGHT = 512;
const FRAME_RATE = 1;
const INITIAL_PROMPT = "a mischievous cat with a third eye, matte pastel colour pallete in a cartoon style";

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
  const processedCanvasRef = useRef<HTMLCanvasElement>(null);
  const socketRef = useRef<WebSocket | null>(null);
  const frameQueueRef = useRef<HTMLImageElement[]>([]);
  const lastWarpedFrameRenderTimeRef = useRef<number | null>(null);
  const isStreamingRef = useRef(true);

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

  // Render frames
  useEffect(() => {
    if (!isRendering) return;

    const renderFrame = () => {
      const now = Date.now();
      const processedCanvas = processedCanvasRef.current;
      if (!processedCanvas) return;

      const processedCtx = processedCanvas.getContext('2d');
      if (!processedCtx) return;

      if (frameQueueRef?.current?.length > 0) {
        const [img, ...remainingFrames] = frameQueueRef.current;
        frameQueueRef.current = remainingFrames;

        if (img) {
          processedCtx.drawImage(img, 0, 0, window.innerWidth, window.innerHeight);
        }
        lastWarpedFrameRenderTimeRef.current = now;
      }
      requestAnimationFrame(renderFrame);
    };

    renderFrame();
  }, [isRendering]);

  // WebSocket connection
  useEffect(() => {
    if (!warp?.podId || warp.podStatus !== 'RUNNING') return;

    const websocketUrl = buildWebsocketUrlFromPodId(warp.podId);
    const socket = new WebSocket(websocketUrl);
    socket.binaryType = 'arraybuffer';

    socket.onmessage = event => {
      const blob = new Blob([event.data], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);
      const img = new Image();
      img.onload = () => {
        URL.revokeObjectURL(url);
        frameQueueRef.current = [...frameQueueRef.current, img];
      };
      img.src = url;
    };

    socketRef.current = socket;

    return () => {
      socket.close();
    };
  }, [warp?.podId, warp?.podStatus]);

  // Send frames
  useEffect(() => {
    if (!currentStream || !socketRef.current) return;

    const videoTrack = currentStream.getVideoTracks()?.[0];
    if (!videoTrack) return;

    const croppedCanvas = croppedCanvasRef.current;
    if (!croppedCanvas) return;

    const croppedCtx = croppedCanvas.getContext('2d');
    if (!croppedCtx) return;

    let animationFrameId: number;

    const sendFrame = async () => {
      if (videoRef.current) {
        croppedCtx.drawImage(videoRef.current, 0, 0, FRAME_WIDTH, FRAME_HEIGHT);
        
        croppedCanvas.toBlob(
          blob => {
            if (blob && socketRef?.current?.readyState === WebSocket.OPEN) {
              blob.arrayBuffer().then(buffer => {
                socketRef?.current?.send(buffer);
              });
            }
          },
          'image/jpeg',
          0.8,
        );
      }
      animationFrameId = requestAnimationFrame(sendFrame);
    };

    sendFrame();

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [currentStream]);

  return (
    <div className="fixed inset-0 bg-black">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        className={`absolute inset-0 w-full h-full object-cover ${frameQueueRef.current.length > 0 ? 'hidden' : ''}`}
      />
      <canvas
        ref={croppedCanvasRef}
        width={FRAME_WIDTH}
        height={FRAME_HEIGHT}
        className="hidden"
      />
      <canvas
        ref={processedCanvasRef}
        width={FRAME_WIDTH}
        height={FRAME_HEIGHT}
        className={`absolute inset-0 w-full h-full object-cover ${frameQueueRef.current.length === 0 ? 'hidden' : ''}`}
      />
    </div>
  );
};

export default WarpPage;
