import time
import sdl2
import sdl2.ext
import numpy as np
import torch
import torch.nn.functional as F
from turbojpeg import TurboJPEG, TJPF_RGB
from threaded_worker import ThreadedWorker
from diffusion_processor import DiffusionProcessor
from settings import Settings
from settings_api import SettingsAPI
from osc_settings_controller import OscSettingsController
from image_utils import (
    unpack_rgb444_image,
    uyvy_to_rgb_batch,
    half_size_batch,
    get_texture_size,
)
import threading
import asyncio
import websockets
from threaded_worker import ThreadedWorker
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from PIL import Image
import signal
import sys
import argparse
import os
import json

from websockets.server import serve
from config import load_server_config


class ThreadedWebsocket(ThreadedWorker):
    def __init__(self, settings):
        super().__init__(has_input=False, has_output=True)
        self.ws_port = settings.websocket_port
        self.jpeg = TurboJPEG()
        self.batch = []
        self.settings_batch = []
        self.batch_size = settings.batch_size
        self.loop = None
        self.settings = settings
        self.server = None
        self.stop_event = threading.Event()
        self.cleanup_called = False
        self.max_queue_size = settings.batch_size * 3
        self.active_connections = set()
        self.authenticated_connections = set()
        self.connection_lock = threading.Lock()
        self.heartbeat_interval = 30
        self.connection_timeout = 90
        self.last_heartbeat = {}
        self.frame_processors = {}  # Track frame processing state per connection
        self.frame_interval = 0.25  # 250ms between frames
        self.batch_interval = self.frame_interval * settings.batch_size
        self.last_batch_time = 0
        self.frame_buffer = []
        self.settings_buffer = []

    async def process_frame(self, websocket, frame_data):
        """Process a single frame from a client"""
        try:
            frame_data_np = np.frombuffer(frame_data, dtype=np.uint8)
            frame = self.jpeg.decode(frame_data_np, pixel_format=TJPF_RGB)
            img = torch.from_numpy(frame).permute(2, 0, 1)
            
            current_time = time.time()
            
            # Add to frame buffer
            self.frame_buffer.append(img.to("cuda"))
            self.settings_buffer.append(self.settings.copy())
            
            # Manage buffer size
            max_buffer_size = self.batch_size * 4  # Store up to 4 batches worth
            if len(self.frame_buffer) > max_buffer_size:
                excess = len(self.frame_buffer) - max_buffer_size
                self.frame_buffer = self.frame_buffer[excess:]
                self.settings_buffer = self.settings_buffer[excess:]

            # Check if it's time to process a batch
            time_since_last_batch = current_time - self.last_batch_time
            if len(self.frame_buffer) >= self.batch_size and time_since_last_batch >= self.batch_interval:
                # Take the most recent batch_size frames
                batch_frames = self.frame_buffer[-self.batch_size:]
                batch_settings = self.settings_buffer[-self.batch_size:]
                
                # Remove the processed frames
                self.frame_buffer = self.frame_buffer[:-self.batch_size]
                self.settings_buffer = self.settings_buffer[:-self.batch_size]
                
                # Process the batch
                batch = torch.stack(batch_frames)
                batch = batch.to(torch.float32) / 255.0
                
                self.output_queue.put((batch, batch_settings))
                self.last_batch_time = current_time

        except Exception as e:
            print(f"Error processing frame: {e}")
            return False
        return True

    async def handler(self, websocket, path):
        """Handle a single WebSocket connection"""
        try:
            # Wait for authentication
            auth_message = await websocket.recv()
            try:
                auth_data = json.loads(auth_message)
                if (auth_data.get("type") == "auth" and 
                    auth_data.get("password") == self.settings.server_password):
                    self.authenticated_connections.add(websocket)
                else:
                    await websocket.close(1008, "Invalid authentication")
                    return
            except json.JSONDecodeError:
                await websocket.close(1008, "Invalid authentication format")
                return

            with self.connection_lock:
                self.active_connections.add(websocket)
                self.last_heartbeat[websocket] = time.time()
                self.frame_processors[websocket] = {
                    'last_frame_time': 0,
                    'frames_processed': 0
                }
            
            print(f"WebSocket connection opened. Active connections: {len(self.active_connections)}")
            
            try:
                async for message in websocket:
                    try:
                        # Handle heartbeat
                        if message == 'pong':
                            self.last_heartbeat[websocket] = time.time()
                            continue

                        # Process frame
                        current_time = time.time()
                        processor_state = self.frame_processors[websocket]
                        
                        # Rate limiting check
                        time_since_last = current_time - processor_state['last_frame_time']
                        if time_since_last < self.frame_interval * 0.9:  # Allow 10% variance
                            continue
                            
                        await self.process_frame(websocket, message)
                        processor_state['last_frame_time'] = current_time
                        processor_state['frames_processed'] += 1

                    except Exception as e:
                        print(f"Error handling message: {e}")
                        await asyncio.sleep(0.1)
                        continue

            except websockets.exceptions.ConnectionClosed:
                print("Connection closed normally")
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            with self.connection_lock:
                self.active_connections.discard(websocket)
                self.authenticated_connections.discard(websocket)
                self.last_heartbeat.pop(websocket, None)
                self.frame_processors.pop(websocket, None)
            print(f"WebSocket connection closed. Remaining connections: {len(self.active_connections)}")

    async def broadcast_to_all(self, data):
        if not self.active_connections:
            return

        disconnected = set()
        for websocket in self.active_connections:
            try:
                await websocket.send(data)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(websocket)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                disconnected.add(websocket)

        # Clean up disconnected clients
        if disconnected:
            with self.connection_lock:
                self.active_connections -= disconnected

    async def send_data(self, data):
        await self.broadcast_to_all(data)

    def setup(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        self.server = self.loop.run_until_complete(
            serve(self.handler, "0.0.0.0", self.ws_port)
        )
        print(f"WebSocket server started on port {self.ws_port}")

    def work(self):
        try:
            self.loop.run_until_complete(self.run_server())
        except Exception as e:
            print(f"Error in ThreadedWebsocket work: {e}")
        finally:
            self.loop.call_soon_threadsafe(self.cleanup)

    async def run_server(self):
        while not self.stop_event.is_set():
            await asyncio.sleep(0.1)

    async def async_cleanup(self):
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        remaining_tasks = [
            task
            for task in asyncio.all_tasks(self.loop)
            if task is not asyncio.current_task()
        ]
        for task in remaining_tasks:
            task.cancel()
        await asyncio.gather(*remaining_tasks, return_exceptions=True)

    def stop_loop(self):
        self.loop.call_soon_threadsafe(self.loop.stop)

    def cleanup(self):
        if self.cleanup_called:
            return
        self.cleanup_called = True

        print("ThreadedWebsocket cleanup")
        if self.loop and not self.loop.is_closed():
            try:
                self.stop_loop()
                future = asyncio.run_coroutine_threadsafe(
                    self.async_cleanup(), self.loop
                )
                try:
                    future.result(timeout=2)  # Wait for up to 2 seconds
                except TimeoutError:
                    print("ThreadedWebsocket async cleanup timed out")
            except Exception as e:
                print(f"Error during ThreadedWebsocket async cleanup: {e}")
            finally:
                if self.loop and not self.loop.is_closed():
                    self.loop.close()

        print("WebSocket server stopped")

    def start(self):
        self.parallel = threading.Thread(target=self.run)
        super().start()

    def close(self):
        print("ThreadedWebsocket closing")
        self.stop_event.set()
        self.loop.call_soon_threadsafe(self.cleanup)
        super().close()


class Processor(ThreadedWorker):
    def __init__(self, settings, use_cached=False):
        super().__init__(has_input=True, has_output=True, debug=True)
        self.batch_size = settings.batch_size
        self.settings = settings
        print("Settings1:", settings)
        self.jpeg = TurboJPEG()
        self.use_cached = use_cached
        self.max_queue_size = settings.batch_size * 2  # Store up to 2 batches worth of frames
        self.diffusion_processor = None

    def setup(self):
        print("🚀 Processor setup starting...")
        self.reload_model()
        self.clear_input()  # drop old frames
        self.runs = 0
        print("✅ Processor setup complete")
    
    def reload_model(self):
        """Reload the diffusion model (for hot-swapping)"""
        # Check if we already have the requested model loaded
        if self.diffusion_processor is not None:
            current_model = getattr(self.diffusion_processor, 'model_type', None)
            if current_model == self.settings.model_type:
                print(f"ℹ️  Model '{self.settings.model_type}' already loaded, skipping reload")
                return
        
        print(f"🔄 ========== RELOADING MODEL ==========")
        print(f"🔄 New model type: {self.settings.model_type}")
        print(f"🔄 Steps: {self.settings.num_inference_steps}, Strength: {self.settings.strength}")
        
        warmup = None
        if self.settings.warmup:
            warmup = self.settings.warmup
            print(f"warmup from settings is: {warmup}")
        
        try:
            # Clean up old model if it exists
            if self.diffusion_processor is not None:
                print("🧹 Cleaning up old model...")
                del self.diffusion_processor
                torch.cuda.empty_cache()
                print("✅ Old model cleaned up")
            
            # Load new model
            print(f"📥 Loading new {self.settings.model_type} model...")
            print(f"📥 Calling DiffusionProcessor.__init__()...")
            self.diffusion_processor = DiffusionProcessor(
                warmup=warmup, use_cached=self.use_cached, settings=self.settings
            )
            print(f"📥 DiffusionProcessor.__init__() returned")
            # Store model type on processor for future checks
            self.diffusion_processor.model_type = self.settings.model_type
            print(f"✅ Model successfully loaded: {self.settings.model_type}")
            print(f"✅ ========== RELOAD COMPLETE ==========")
        except Exception as e:
            print(f"❌ Error reloading model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def work(self, args):
        # Log first call
        if not hasattr(self, '_processor_work_count'):
            self._processor_work_count = 0
            print(f"🔧 Processor.work() first call! Args type: {type(args)}")
        self._processor_work_count += 1
        
        if self._processor_work_count % 10 == 0:
            print(f"🔧 Processor.work() called {self._processor_work_count} times")
        
        # Check if diffusion processor is ready
        if self.diffusion_processor is None:
            print("⚠️ Diffusion processor not ready yet, skipping frame")
            return
        
        # Process only if we're not too backed up
        while self.input_queue.qsize() > self.max_queue_size:
            # Drop old frames if we're getting backed up
            try:
                _ = self.input_queue.get_nowait()
            except:
                break

        images, settings_batch = args

        try:
            results = self.diffusion_processor.run(
                images=images,
                prompt=self.settings.prompt,
                use_compel=True,
                num_inference_steps=self.settings.num_inference_steps,
                strength=self.settings.strength,
                seed=self.settings.seed,
            )

            # Log frame generation
            if not hasattr(self, '_frame_gen_count'):
                self._frame_gen_count = 0
                print(f"🎨 Processor: First batch of {len(results)} frames generated!")
                
            for frame_settings, image, result in zip(settings_batch, images, results):
                # Convert to uint8 and encode directly (no upscaling)
                result_uint8 = (result * 255).astype(np.uint8)
                # Use quality setting for compression
                result_bytes = self.jpeg.encode(result_uint8, pixel_format=TJPF_RGB, quality=self.settings.jpeg_quality)
                self.output_queue.put(result_bytes)
                
                self._frame_gen_count += 1
                
            if self._frame_gen_count % 30 == 0:
                print(f"🎨 Processor: {self._frame_gen_count} frames → output_queue (queue size: {self.output_queue.qsize()})")
                
        except Exception as e:
            print(f"❌ Error in diffusion processing: {e}")
            import traceback
            traceback.print_exc()
        
        self.runs += 1
        if self.runs < 3:
            print("warming up, dropping old frames")
            self.clear_input()


class BroadcastStream(ThreadedWorker):
    def __init__(self, port, settings, threaded_websocket):
        super().__init__(has_input=True, has_output=False)
        self.port = port
        self.fullscreen = False
        self.settings = settings
        self.threaded_websocket = threaded_websocket
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.batch_size = settings.batch_size
        self.last_snapshot_time = 0
        self.snapshot_counter = 0

    def setup(self):
        self.jpeg = TurboJPEG()
        
        # Create snapshot directory if it doesn't exist
        if self.settings.save_snapshots:
            os.makedirs(self.settings.snapshot_directory, exist_ok=True)
            print(f"Snapshot directory ready: {self.settings.snapshot_directory}")
        

    def broadcast_msg(self, jpg):
        # Log first call
        if not hasattr(self, '_broadcast_count'):
            self._broadcast_count = 0
        self._broadcast_count += 1
        
        if self._broadcast_count % 30 == 0:
            print(f"📡 broadcast_msg called {self._broadcast_count} times")
        
        try:
            # Send to WebSocket clients (legacy/fallback support)
            if self.threaded_websocket is not None:
                future = asyncio.run_coroutine_threadsafe(
                    self.threaded_websocket.send_data(jpg), self.threaded_websocket.loop
                )
                future.result()  # Wait for the coroutine to complete
            
        except Exception as e:
            print(f"Error in broadcast_msg: {e}")

    def save_snapshot(self, frame):
        """Save a snapshot of the current frame"""
        try:
            from datetime import datetime
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}_{self.snapshot_counter:04d}.jpg"
            filepath = os.path.join(self.settings.snapshot_directory, filename)
            
            # Save the JPEG data to file
            with open(filepath, 'wb') as f:
                f.write(frame)
            
            self.snapshot_counter += 1
            print(f"📸 Snapshot saved: {filepath}")
            
        except Exception as e:
            print(f"Error saving snapshot: {e}")
    
    def work(self, frame):
        # Log first call
        if not hasattr(self, '_work_count'):
            self._work_count = 0
            print("🎬 BroadcastStream.work() starting...")
        self._work_count += 1
        
        if self._work_count == 1:
            print(f"📦 First frame received in BroadcastStream! Type: {type(frame)}, Size: {len(frame) if isinstance(frame, bytes) else 'unknown'}")
        
        if self._work_count % 30 == 0:
            print(f"🎬 BroadcastStream.work() called {self._work_count} times, queue size: {self.input_queue.qsize()}")
        
        try:
            while self.input_queue.qsize() > self.settings.batch_size:
                frame = self.input_queue.get()

            # Always broadcast (to both WebSocket and WebRTC)
            self.broadcast_msg(frame)
            
            # Check if it's time to save a snapshot
            if self.settings.save_snapshots:
                current_time = time.time()
                if current_time - self.last_snapshot_time >= self.settings.snapshot_interval:
                    self.save_snapshot(frame)
                    self.last_snapshot_time = current_time
                    
        except Exception as e:
            print(f"Error in work: {e}")

    def cleanup(self):
        try:
            if hasattr(self, "texture") and self.texture is not None:
                sdl2.SDL_DestroyTexture(self.texture)
            sdl2.ext.quit()
        except Exception as e:
            print(f"Error during cleanup: {e}")


class GenDJ:
    def __init__(self):
        self.settings = Settings()
        self.websocket = ThreadedWebsocket(self.settings)
        
        # Create processor first, then pass it to settings_api for model hot-swapping
        # Note: This is a simplified version - full implementation would need proper processor
        self.settings_api = SettingsAPI(self.settings)  # No processor reference in simple version
        
        self.diffusion = DiffusionProcessor(self.settings)
        self.running = True
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, signum, frame):
        print("\nReceived shutdown signal...")
        self.running = False
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        print("Starting cleanup...")
        self.settings_api.stop()
        self.websocket.stop()
        self.diffusion.stop()
        print("Waiting for components to shut down...")
        self.settings_api.close()
        self.websocket.close()
        self.diffusion.close()
        print("Cleanup complete")

    def run(self):
        try:
            self.settings_api.start()
            self.websocket.start()
            self.diffusion.start()
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt...")
            self.cleanup()
        except Exception as e:
            print(f"Error in main loop: {e}")
            self.cleanup()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run gendj.py with specified options.")
    parser.add_argument(
        "--use_cached",
        action="store_true",
        help="Use cached models in DiffusionProcessor",
    )
    args = parser.parse_args()

    settings = Settings()
    settings_controller = OscSettingsController(settings)

    receiver = ThreadedWebsocket(settings)
    processor = Processor(settings, use_cached=args.use_cached).feed(receiver)
    
    # Create settings API with processor reference for model hot-swapping
    settings_api = SettingsAPI(settings, processor=processor)
    
    # Create display for broadcasting
    display = BroadcastStream(settings.output_port, settings, receiver).feed(processor)
    
    # Verify pipeline connections
    print(f"🔗 Pipeline: receiver → processor → display")
    print(f"🔗 processor.input_queue is receiver.output_queue: {processor.input_queue is receiver.output_queue}")
    print(f"🔗 display.input_queue is processor.output_queue: {display.input_queue is processor.output_queue}")

    # Main program signal handling
    def signal_handler(signal, frame):
        print("Signal received, closing...")
        components = [display, processor, receiver, settings_controller, settings_api]

        for component in components:
            component_name = getattr(component, "name", component.__class__.__name__)
            print(f"Closing {component_name}...")
            if hasattr(component, "close"):
                component.close()

        # Wait for all components to finish
        for component in components:
            if hasattr(component, "parallel"):
                try:
                    component.parallel.join(timeout=10)
                except TimeoutError:
                    print(f"{component.__class__.__name__} failed to close in time")

        print("All components closed, exiting...")
        os._exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start the components
    print("🚀 ========== STARTING COMPONENTS ==========")
    print("🚀 Starting settings_api...")
    settings_api.start()
    print("🚀 Starting settings_controller...")
    settings_controller.start()
    print("🚀 Starting display (BroadcastStream)...")
    display.start()
    print("🚀 Starting processor (Processor)...")
    processor.start()
    print("🚀 Starting receiver (ThreadedWebsocket)...")
    receiver.start()
    print("✅ ========== ALL COMPONENTS STARTED ==========")
    print("👀 Waiting for webcam frames...")

    exit_event = threading.Event()
    try:
        while not exit_event.is_set():
            exit_event.wait(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("Main loop exiting, closing components...")
        signal_handler(signal.SIGINT, None)


if __name__ == "__main__":
    config = load_server_config()
    print(f"Server password configured: {bool(config.SERVER_PASSWORD)}")
    main()
