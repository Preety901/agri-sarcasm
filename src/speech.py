import io
import numpy as np
from scipy.io import wavfile

_asr = {}

def _load(model):
    if model not in _asr:
        from transformers import pipeline
        _asr[model] = pipeline("automatic-speech-recognition", model=model)
    return _asr[model]

def transcribe(wav_bytes, model="openai/whisper-base", language=None):
    """Speech -> text. language=None lets Whisper auto-detect; or pass 'hindi' / 'english'."""
    asr = _load(model)
    sr, audio = wavfile.read(io.BytesIO(wav_bytes))
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if np.abs(audio).max() > 1.5:
        audio = audio / 32768.0
    if sr != 16000:  # simple resample, no extra dependency
        n = int(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    gk = {"task": "transcribe"}
    if language:
        gk["language"] = language
    out = asr({"raw": audio, "sampling_rate": 16000}, generate_kwargs=gk)
    return out["text"].strip()
