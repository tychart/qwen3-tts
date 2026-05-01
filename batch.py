import argparse
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel


MODELS = {
    "small": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "big": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

def get_file_from_file_path(file_path: str) -> str:
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content

# print(f"Ref text: {get_file_from_file_path('skyward_30s_segment_suzy.txt')}")


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

    # print(f"Ref text: {get_file_from_file_path('skyward_30s_segment_suzy.txt')}")

    voice_clone_prompt_suzy = model.create_voice_clone_prompt(
        ref_audio="skyward_30s_segment_suzy.wav",
        ref_text=get_file_from_file_path("skyward_30s_segment_suzy.txt"),
        x_vector_only_mode=False,  
    )

    print(f"Generating speech for: {text}")
    
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=args.language,
        voice_clone_prompt=voice_clone_prompt_suzy,
    )

    sf.write(args.output, wavs[0], sr)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()