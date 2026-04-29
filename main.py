import argparse
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


MODELS = {
    "small": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "big": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Inference")
    parser.add_argument(
        "--model",
        choices=["small", "big"],
        default="small",
        help="Model size: small (0.6B) or big (1.7B)",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to synthesize (use --text-file for file input)",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        help="File containing text to synthesize",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Language of the text",
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Ryan",
        help="Speaker voice to use",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output audio file path",
    )
    args = parser.parse_args()

    if args.text_file:
        with open(args.text_file, "r") as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        parser.error("Either --text or --text-file must be provided")

    model_id = MODELS[args.model]
    print(f"Loading {args.model} model: {model_id}")

    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    print(f"Generating speech for: {text}")
    wavs, sr = model.generate_custom_voice(
        text=text,
        language=args.language,
        speaker=args.speaker,
    )

    sf.write(args.output, wavs[0], sr)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()