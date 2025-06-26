import subprocess
import os
import torch
import torchaudio
from transformers import AutoProcessor, WhisperForConditionalGeneration
from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
import re
from crewai.tools import BaseTool
from config import llm  # Assuming llm is an instance of your LLM model


# Load Whisper model and processor once to avoid reloading on every call
model_name = "openai/whisper-base"
processor = AutoProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # type: ignore

def download_youtube_audio(youtube_url: str, output_path: str = "temp_audio.%(ext)s") -> str:
    """Download audio from YouTube video using yt-dlp"""
    command = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "-o", output_path,
        youtube_url
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download audio: {str(e)}")

    downloaded_file = output_path.replace("%(ext)s", "wav")
    return downloaded_file

def transcribe_audio_with_whisper(audio_path: str) -> str:
    """Transcribe audio using Whisper model from Hugging Face"""
    waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Prepare input for Whisper
    input_features = processor(
        waveform.squeeze().numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)

    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def youtube_transcribe_tool(youtube_url: str) -> str:
    """Main function to transcribe YouTube video. Tries youtube-transcript-api first, falls back to Whisper if needed."""
    try:
        # Extract video ID from URL
        match = re.search(r"v=([\w-]+)", youtube_url)
        if not match:
            return "Invalid YouTube URL."
        video_id = match.group(1)
        # Try to get transcript using youtube-transcript-api
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_text = " ".join([entry['text'] for entry in transcript])
        return full_text
    except Exception as e:
        print(f"youtube-transcript-api failed: {e}. Falling back to Whisper.")
        try:
            print("🔽 Downloading audio...")
            audio_path = download_youtube_audio(youtube_url)

            print("🎤 Transcribing audio...")
            transcript = transcribe_audio_with_whisper(audio_path)

            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as cleanup_error:
                print(f"Warning: Failed to delete audio file {audio_path}: {cleanup_error}")

            print("✅ Transcription complete.")
            return transcript
        except Exception as e2:
            return f"Error transcribing video: {str(e2)}"

# Fix the error by properly initializing the BaseTool class
class YouTubeTranscriberTool(BaseTool):
    def __init__(self, name="YouTube Transcriber", description="Transcribes a YouTube video using transcript API or Whisper as fallback."):
        # Initialize the base class with required arguments
        super().__init__(name=name, description=description)

    def _run(self, youtube_url: str) -> str:
        return youtube_transcribe_tool(youtube_url)

# Now the tool is correctly initialized
youtube_transcriber_tool = YouTubeTranscriberTool()
