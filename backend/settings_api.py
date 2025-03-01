import threading
import uvicorn
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import aiofiles
import os
import tempfile

import time
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from safety_checker import SafetyChecker
from speech_processor import SpeechProcessor


class SettingsAPI:
    def __init__(self, settings):
        self.shutdown = False
        self.settings = settings
        port = settings.settings_port
        self.app = FastAPI()
        self.setup_routes()
        self.thread = threading.Thread(target=self.run_server, args=(port,))
        # Add a thread pool with thread naming for better cleanup
        self.executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="audio_processor"
        )
        self._transcribing = False  # Lock for transcription
        self._transcribe_lock = threading.Lock()
        self._server = None  # Store server instance
        self.prompt_0 = settings.prompt
        self.prompt_1 = "A psychedelic landscape."
        self.blend = 0
        self.speech_processor = SpeechProcessor(device="cuda")
        self.base_prompt = "You are Mischief, a magic mirror at a party. You listen to words in your surroundings and create a fun visual based on the words - all in a fun pastel cartoon style. You've recently heard people nearby say:"
        print("Speech processor initialized")

    def update_blend(self):
        if self.blend == 0:
            self.settings.prompt = self.prompt_0
        elif self.blend == 1:
            self.settings.prompt = self.prompt_1
        else:
            a = self.prompt_0
            b = self.prompt_1
            t = self.blend
            self.settings.prompt = f'("{a}", "{b}").blend({1-t:.2f}, {t:.2f})'

    def start(self):
        print("SettingsAPI starting")
        if not self.thread.is_alive():
            self.thread.start()

    def setup_routes(self):
        app = self.app
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.post("/prompt/{msg}")
        async def prompt(msg: str):
            prompt = msg

            override = "-f" in prompt
            if override:
                prompt = prompt.replace("-f", "").strip()
            if self.settings.safety and not override:
                safety_checker = SafetyChecker()
                safety = safety_checker(prompt)
                if safety != "safe":
                    print(f"Ignoring prompt ({safety}):", prompt)
                    return {"safety": "unsafe"}

            self.prompt_0 = prompt
            self.update_blend()
            print("Updated prompt:", prompt)
            return {"safety": "safe"}

        @app.post("/secondprompt/{msg}")
        async def secondprompt(msg: str):
            prompt = msg

            override = "-f" in prompt
            if override:
                prompt = prompt.replace("-f", "").strip()
            if self.settings.safety and not override:
                safety_checker = SafetyChecker()
                safety = safety_checker(prompt)
                if safety != "safe":
                    print(f"Ignoring prompt ({safety}):", prompt)
                    return {"safety": "unsafe"}

            self.prompt_1 = prompt
            self.update_blend()
            print("Updated secondprompt:", prompt)
            return {"safety": "safe"}

        @app.post("/blend/{msg}")
        async def blend(msg: str):
            try:
                blend_value = float(msg)
                if 0 <= blend_value <= 1:
                    self.blend = blend_value
                    self.update_blend()
                    return {"status": "success", "blend": self.blend}
                else:
                    return {
                        "status": "error",
                        "message": "Blend value must be between 0 and 1",
                    }
            except ValueError:
                return {"status": "error", "message": "Invalid blend value"}

        @app.get("/directory/{status}")
        async def directory(status: str):
            self.settings.directory = "data/" + status
            print("Updated directory status:", self.settings.directory)
            return {"status": "updated"}

        @app.get("/debug/{status}")
        async def debug(status: bool):
            self.settings.debug = status
            print("Updated debug status:", status)
            return {"status": "updated"}

        @app.get("/compel/{status}")
        async def compel(status: bool):
            self.settings.compel = status
            print("Updated compel status:", status)
            return {"status": "updated"}

        @app.get("/passthrough/{status}")
        async def passthrough(status: bool):
            self.settings.passthrough = status
            print("Updated passthrough status:", self.settings.passthrough)
            return {"status": "updated"}

        @app.get("/fixed_seed/{status}")
        async def fixed_seed(status: bool):
            self.settings.fixed_seed = status
            print("Updated fixed_seed status:", self.settings.fixed_seed)
            return {"status": "updated"}

        @app.get("/mirror/{status}")
        async def mirror(status: bool):
            self.settings.mirror = status
            print("Updated mirror status:", status)
            return {"status": "updated"}

        @app.get("/batch_size/{value}")
        async def batch_size(value: int):
            self.settings.batch_size = value
            print("Updated batch_size:", self.settings.batch_size)
            return {"status": "updated"}

        @app.get("/seed/{value}")
        async def seed(value: int):
            self.settings.seed = value
            print("Updated seed:", self.settings.seed)
            return {"status": "updated"}

        @app.get("/steps/{value}")
        async def steps(value: int):
            self.settings.num_inference_steps = value
            print("Updated num_inference_steps:", self.settings.num_inference_steps)
            return {"status": "updated"}

        @app.get("/strength/{value}")
        async def strength(value: float):
            self.settings.strength = value
            print("Updated strength:", self.settings.strength)
            return {"status": "updated"}

        @app.get("/opacity/{value}")
        async def opacity(value: float):
            value = min(max(value, 0), 1)
            self.settings.opacity = value
            print("Updated opacity:", self.settings.opacity)
            return {"status": "updated"}

        @app.post("/transcribe")
        async def transcribe_audio(audio: UploadFile = File(...)):
            # Check if we're already transcribing
            if self._transcribing:
                print("⚠️ Already processing audio, skipping new request")
                return {"error": "Already processing audio"}

            print("Transcribing audio...")
            try:
                # Acquire transcription lock
                with self._transcribe_lock:
                    if self._transcribing:
                        return {"error": "Already processing audio"}
                    self._transcribing = True

                content = await audio.read()
                print(f"Received audio data of size: {len(content)} bytes")
                
                # Use our dedicated executor with timeout
                future = self.executor.submit(self.speech_processor.process_audio, content)
                try:
                    transcribed_text = await asyncio.get_event_loop().run_in_executor(
                        None, future.result, 300  # 30 second timeout
                    )
                except TimeoutError:
                    print("❌ Audio processing timed out")
                    self._transcribing = False
                    return {"error": "Audio processing timed out"}
                
                if transcribed_text:
                    print(f"🎤 Transcribed: '{transcribed_text}'")
                    
                    # Generate and set new prompt
                    new_prompt = f"{self.base_prompt} '{transcribed_text}', transport your unexpecting subjects there"
                    print(f"🔄 Setting new prompt: {new_prompt}")
                    self.settings.prompt = new_prompt
                    
                    self._transcribing = False
                    return {"text": transcribed_text}
                else:
                    print("❌ No text transcribed from audio")
                    self._transcribing = False
                    return {"error": "Failed to transcribe audio"}
                    
            except Exception as e:
                print(f"❌ Error transcribing audio: {e}")
                import traceback
                traceback.print_exc()
                self._transcribing = False
                return {"error": str(e)}

        if "READY_WEBHOOK_URL" not in os.environ:
            app.mount("/", StaticFiles(directory="fe", html=True), name="static")

    def run_server(self, port):
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        config = uvicorn.Config(
            self.app,
            host="0.0.0.0",
            port=port,
            log_level="info",
            loop="asyncio",
            access_log=True
        )
        self._server = uvicorn.Server(config=config)
        try:
            loop.run_until_complete(self._server.serve())
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            loop.close()

    def stop(self):
        self.shutdown = True
        print("Stopping SettingsAPI...")
        # First stop accepting new tasks
        self.executor.shutdown(wait=False)
        print("Shutting down audio processing executor...")
        
        # Force kill any running audio processing threads
        for thread in threading.enumerate():
            if thread.name.startswith('audio_processor'):
                print(f"Forcing audio thread shutdown: {thread.name}")
                # Force the thread to stop
                try:
                    import ctypes
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(thread.ident),
                        ctypes.py_object(SystemExit)
                    )
                except Exception as e:
                    print(f"Error forcing thread shutdown: {e}")

        if self._server:
            self._server.should_exit = True
            print("Server shutdown signal sent")

    def close(self):
        print("SettingsAPI closing")
        self.stop()  # Ensure stop is called
        
        # Wait for a short time for graceful shutdown
        shutdown_timeout = 2  # seconds
        try:
            self.thread.join(timeout=shutdown_timeout)
        except TimeoutError:
            print("Server thread failed to stop gracefully")
        
        if self.thread.is_alive():
            print("Server thread still alive, forcing exit...")
            # Force exit if still running
            import os
            import signal
            os.kill(os.getpid(), signal.SIGTERM)

        print("SettingsAPI closed")
