import torch
import torchaudio as ta

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS

device = "cuda" if torch.cuda.is_available() else "cpu"

text = "안녕하세요. 파인튜닝된 모델을 기반으로 생성된 한국어 음성입니다."
voice_prompt = "prompt.wav"

def load_finetuned_or_fallback():
    try:
        tts = ChatterboxTTS.from_pretrained(device=device)
        tts.load_state_dict(torch.load("final.pt", map_location=device), strict=False)
        return tts
    except Exception:
        return ChatterboxMultilingualTTS.from_pretrained(device=device)

model = load_finetuned_or_fallback()

with torch.no_grad():
    if isinstance(model, ChatterboxMultilingualTTS):
        wav = model.generate(
            text=text,
            language_id="ko",
            audio_prompt_path=voice_prompt
        )
        sr = model.sr
    else:
        wav = model.generate(
            text=text,
            audio_prompt_path=voice_prompt
        )
        sr = model.sr

ta.save("output_ko.wav", wav.cpu(), sr)
