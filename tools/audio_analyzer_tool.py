from crewai_tools import BaseTool
import torch
import torchaudio
from transformers import AutoProcessor, WhisperForConditionalGeneration

class LocalAudioTranscriberTool(BaseTool):
    name = "Local Audio Transcriber"
    description = "Transcribes a local .wav audio file using Whisper"

    def _run(self, audio_path: str) -> str:
        try:
            model_name = "openai/whisper-base"
            processor = AutoProcessor.from_pretrained(model_name)
            model = WhisperForConditionalGeneration.from_pretrained(model_name)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)

            waveform, sample_rate = torchaudio.load(audio_path)

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sample_rate != 16000:
                waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

            input_features = processor(
                waveform.squeeze().numpy(),
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(device)

            with torch.no_grad():
                predicted_ids = model.generate(input_features)

            return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        except Exception as e:
            return f"❌ Error: {str(e)}"
