"""Chatterbox TTS inference script for Streamlit integration."""

import argparse
import torch
import torchaudio as ta
from pathlib import Path

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS


def load_finetuned_model(device: str, checkpoint_path: str = None):
    """
    Load finetuned model if checkpoint provided, otherwise fall back to multilingual.

    Following the pattern from inference_semantic.py which works.
    """
    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            print(f"Loading finetuned ChatterboxTTS with checkpoint: {checkpoint_path}")
            tts = ChatterboxTTS.from_pretrained(device=device)
            tts.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
            print("Finetuned checkpoint loaded successfully")
            return tts, "finetuned"
        except Exception as e:
            print(f"Warning: Failed to load finetuned checkpoint: {e}")
            print("Falling back to ChatterboxMultilingualTTS...")

    print("Loading ChatterboxMultilingualTTS...")
    return ChatterboxMultilingualTTS.from_pretrained(device=device), "multilingual"


def main():
    parser = argparse.ArgumentParser(description="Chatterbox TTS Inference")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--text-file", type=str, default=None, help="Read text from file instead")
    parser.add_argument("--voice-prompt", type=str, default=None, help="Voice prompt audio file for cloning")
    parser.add_argument("--output", type=str, required=True, help="Output audio file path")
    parser.add_argument("--language", type=str, default="ko", help="Language ID (e.g., ko, en, ja, zh)")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Exaggeration parameter (0.0-1.0)")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight parameter (0.0-1.0)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to finetuned checkpoint")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Get text from file or argument
    if args.text_file:
        with open(args.text_file, "r") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        raise ValueError("Either --text or --text-file must be provided")

    print(f"Text: {text}")
    print(f"Language: {args.language}")
    print(f"Voice prompt: {args.voice_prompt}")
    print(f"Output: {args.output}")
    print(f"Exaggeration: {args.exaggeration}")
    print(f"CFG Weight: {args.cfg_weight}")

    # Load model (finetuned if available, otherwise multilingual)
    model, model_type = load_finetuned_model(device, args.checkpoint)

    # Generate audio
    print("Generating audio...")
    with torch.no_grad():
        generate_kwargs = {
            "text": text,
            "exaggeration": args.exaggeration,
            "cfg_weight": args.cfg_weight,
        }

        # Add voice prompt if provided
        if args.voice_prompt and Path(args.voice_prompt).exists():
            generate_kwargs["audio_prompt_path"] = args.voice_prompt
            print(f"Using voice prompt: {args.voice_prompt}")

        # Add language_id only for multilingual model
        if model_type == "multilingual":
            generate_kwargs["language_id"] = args.language

        wav = model.generate(**generate_kwargs)

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(output_path), wav.cpu(), model.sr)
    print(f"Audio saved to: {output_path}")


if __name__ == "__main__":
    main()
