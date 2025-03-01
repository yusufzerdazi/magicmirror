import torch
import torchaudio
from faster_whisper import WhisperModel
import numpy as np
import io
import subprocess
import tempfile
import os

class SpeechProcessor:
    def __init__(self, device="cuda"):
        # Force CPU for transcription to avoid CUDA/cuDNN issues
        self.device = "cpu"
        # Initialize Whisper model on GPU with FasterWhisper implementation
        self.model = WhisperModel(
            model_size_or_path="base",
            device="cpu",  # Use CPU
            compute_type="float32"  # Use float32 for CPU
        )
        self.last_transcription = None
        self.min_confidence = 0.5  # Minimum confidence threshold
        print("Speech recognition model loaded on CPU")

    def is_repetitive(self, text):
        """Check if text contains repetitive patterns"""
        # Convert to lowercase and split into words
        words = text.lower().split()
        if len(words) < 4:  # Too short to be repetitive
            return False
        
        # Check for exact repetition of the last transcription
        if self.last_transcription and text.lower() == self.last_transcription.lower():
            return True

        # Check for word-level repetition patterns
        for i in range(1, len(words)//2):
            pattern = words[:i]
            repetitions = [words[j:j+i] for j in range(0, len(words), i)]
            if len(repetitions) >= 3 and all(r == pattern for r in repetitions[:-1]):
                return True
                
        return False

    def convert_webm_to_wav(self, webm_data):
        webm_file = None
        wav_file = None
        try:
            # Create temporary files for conversion
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as webm_file, \
                 tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                
                # Write webm data to temp file
                webm_file.write(webm_data)
                webm_file.flush()
                
                # Convert webm to wav using ffmpeg
                result = subprocess.run([
                    'ffmpeg',
                    '-loglevel', 'error',
                    '-y',
                    '-i', webm_file.name,
                    '-ar', '16000',
                    '-ac', '1',
                    '-f', 'wav',
                    wav_file.name
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"FFmpeg conversion failed: {result.stderr}")
                
                # Read the converted wav file
                waveform, sample_rate = torchaudio.load(wav_file.name)
                # Move to CPU explicitly if needed
                if waveform.device.type != "cpu":
                    waveform = waveform.cpu()
                
                return waveform, sample_rate
        finally:
            # Clean up temp files
            try:
                if webm_file and os.path.exists(webm_file.name):
                    os.unlink(webm_file.name)
                if wav_file and os.path.exists(wav_file.name):
                    os.unlink(wav_file.name)
            except OSError as e:
                print(f"Warning: Failed to clean up temp files: {e}")

    def process_audio(self, audio_data, sample_rate=16000):
        try:
            print("Starting audio processing...")
            # Convert webm to wav
            waveform, sample_rate = self.convert_webm_to_wav(audio_data)
            print(f"Audio converted to wav: shape={waveform.shape}, sample_rate={sample_rate}")
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                print("Converted stereo to mono")

            # Convert to numpy array
            audio_np = waveform.squeeze().numpy()
            print(f"Converted to numpy array: shape={audio_np.shape}")

            # Transcribe with Whisper
            print("Starting transcription...")
            segments, info = self.model.transcribe(
                audio_np,
                beam_size=5,
                language="en",
                temperature=0.0
            )

            # Combine all segments and check confidence
            text = ""
            avg_confidence = 0.0
            segment_count = 0
            
            for segment in segments:
                if segment.avg_logprob > -1:  # Filter out very low confidence segments
                    text += " " + segment.text
                    avg_confidence += np.exp(segment.avg_logprob)  # Convert log prob to probability
                    segment_count += 1
            
            if segment_count > 0:
                text = text.strip()
                avg_confidence = avg_confidence / segment_count
                
                print(f"Transcription complete: '{text}' (confidence: {avg_confidence:.2f})")
                
                # Check for repetitive patterns and confidence threshold
                if self.is_repetitive(text):
                    print("❌ Rejected: Repetitive text detected")
                    return None
                
                self.last_transcription = text
                return text.strip()
            else:
                print("❌ No valid segments found")
                return None

        except Exception as e:
            print(f"❌ Error processing audio: {e}")
            import traceback
            traceback.print_exc()
            return None 