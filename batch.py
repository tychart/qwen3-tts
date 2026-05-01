import argparse
import re
import statistics
import time
from typing import Iterable, List

import torch
from qwen_tts import Qwen3TTSModel


MODELS = {
    "small": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "big": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}

DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
REFERENCE_AUDIO = "skyward_30s_segment_suzy.wav"
REFERENCE_TEXT = "skyward_30s_segment_suzy.txt"
BYTES_PER_GIB = 1024 ** 3


def read_text_file(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


def parse_batch_sizes(value: str) -> List[int]:
    batch_sizes = []

    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue

        try:
            batch_size = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid batch size {part!r}; expected a comma-separated list of integers."
            ) from exc

        if batch_size <= 0:
            raise argparse.ArgumentTypeError(
                f"Invalid batch size {batch_size}; values must be positive integers."
            )

        batch_sizes.append(batch_size)

    if not batch_sizes:
        raise argparse.ArgumentTypeError("At least one batch size must be provided.")

    return batch_sizes


def split_into_sentences(text: str) -> List[str]:
    chunks = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]
    return chunks


def batched(items: List[str], batch_size: int) -> Iterable[List[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def is_cuda_oom_error(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def clear_cuda_after_failure() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def maybe_synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gib_from_bytes(byte_count: int) -> float:
    return byte_count / BYTES_PER_GIB


def format_gib(value: float) -> str:
    return f"{value:.2f} GiB"


def build_voice_clone_prompt(model: Qwen3TTSModel):
    return model.create_voice_clone_prompt(
        ref_audio=REFERENCE_AUDIO,
        ref_text=read_text_file(REFERENCE_TEXT),
        x_vector_only_mode=False,
    )


def run_generation_pass(
    model: Qwen3TTSModel,
    sentences: List[str],
    batch_size: int,
    language: str,
    base_prompt,
) -> float:
    total_audio_seconds = 0.0

    for sentence_batch in batched(sentences, batch_size):
        prompt_batch = base_prompt * len(sentence_batch)
        wavs, sample_rate = model.generate_voice_clone(
            text=sentence_batch,
            language=language,
            voice_clone_prompt=prompt_batch,
        )
        total_audio_seconds += sum(len(wav) / sample_rate for wav in wavs)

    return total_audio_seconds


def benchmark_batch_size(
    model: Qwen3TTSModel,
    sentences: List[str],
    batch_size: int,
    language: str,
    base_prompt,
) -> dict:
    total_chars = sum(len(sentence) for sentence in sentences)

    clear_cuda_after_failure()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    maybe_synchronize_cuda()
    start = time.perf_counter()
    total_audio_seconds = run_generation_pass(
        model=model,
        sentences=sentences,
        batch_size=batch_size,
        language=language,
        base_prompt=base_prompt,
    )
    maybe_synchronize_cuda()
    elapsed = time.perf_counter() - start

    peak_allocated_gib = 0.0
    peak_reserved_gib = 0.0
    if torch.cuda.is_available():
        peak_allocated_gib = gib_from_bytes(torch.cuda.max_memory_allocated())
        peak_reserved_gib = gib_from_bytes(torch.cuda.max_memory_reserved())

    return {
        "batch_size": batch_size,
        "total_sentences": len(sentences),
        "total_chars": total_chars,
        "total_audio_seconds": total_audio_seconds,
        "wall_time_seconds": elapsed,
        "audio_seconds_per_second": total_audio_seconds / elapsed if elapsed else 0.0,
        "chars_per_second": total_chars / elapsed if elapsed else 0.0,
        "sentences_per_second": len(sentences) / elapsed if elapsed else 0.0,
        "peak_allocated_gib": peak_allocated_gib,
        "peak_reserved_gib": peak_reserved_gib,
    }


def average_run_metrics(runs: List[dict]) -> dict:
    keys_to_average = (
        "total_audio_seconds",
        "wall_time_seconds",
        "audio_seconds_per_second",
        "chars_per_second",
        "sentences_per_second",
        "peak_allocated_gib",
        "peak_reserved_gib",
    )

    averaged = {
        "batch_size": runs[0]["batch_size"],
        "total_sentences": runs[0]["total_sentences"],
        "total_chars": runs[0]["total_chars"],
    }

    for key in keys_to_average:
        averaged[key] = statistics.mean(run[key] for run in runs)

    return averaged


def format_summary_line(metrics: dict, benchmark_runs: int) -> str:
    return (
        f"batch={metrics['batch_size']:<2} "
        f"runs={benchmark_runs} "
        f"sentences={metrics['total_sentences']:<3} "
        f"chars={metrics['total_chars']:<5} "
        f"audio_s={metrics['total_audio_seconds']:.2f} "
        f"time_s={metrics['wall_time_seconds']:.2f} "
        f"audio_s_per_s={metrics['audio_seconds_per_second']:.2f} "
        f"chars_per_s={metrics['chars_per_second']:.2f} "
        f"sent_per_s={metrics['sentences_per_second']:.2f} "
        f"peak_alloc={format_gib(metrics['peak_allocated_gib'])} "
        f"peak_reserved={format_gib(metrics['peak_reserved_gib'])}"
    )


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be 0 or greater")
    if args.benchmark_runs <= 0:
        parser.error("--benchmark-runs must be greater than 0")
    if args.max_sentences is not None and args.max_sentences <= 0:
        parser.error("--max-sentences must be greater than 0")


def load_input_text(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    if args.text_file:
        return read_text_file(args.text_file).strip()
    if args.text:
        return args.text.strip()
    parser.error("Either --text or --text-file must be provided")
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark native batch processing for Qwen3 voice-clone TTS."
    )
    parser.add_argument(
        "--model",
        choices=["small", "big"],
        default="small",
        help="Model size: small (0.6B) or big (1.7B).",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to synthesize (use --text-file for file input).",
    )
    parser.add_argument(
        "--text-file",
        type=str,
        help="File containing text to synthesize.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="English",
        help="Language of the text.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=parse_batch_sizes,
        default=list(DEFAULT_BATCH_SIZES),
        help="Comma-separated list of batch sizes to benchmark. Default: 1,2,4,8.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of untimed warmup runs per batch size.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="Number of timed benchmark runs per batch size.",
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        help="Optional cap on sentence chunks for quicker tests.",
    )
    args = parser.parse_args()

    validate_args(args, parser)

    input_text = load_input_text(args, parser)
    sentences = split_into_sentences(input_text)
    if args.max_sentences is not None:
        sentences = sentences[:args.max_sentences]

    if not sentences:
        parser.error("No sentence chunks were produced from the input text.")

    model_id = MODELS[args.model]
    total_chars = sum(len(sentence) for sentence in sentences)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    print(f"Loading {args.model} model: {model_id}")
    model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    print("Building reusable voice-clone prompt...")
    base_prompt = build_voice_clone_prompt(model)

    print("Benchmark configuration")
    print(f"  GPU: {device_name}")
    print(f"  Sentences: {len(sentences)}")
    print(f"  Characters: {total_chars}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Warmup runs: {args.warmup_runs}")
    print(f"  Benchmark runs: {args.benchmark_runs}")
    print()

    successful_results = []

    for batch_size in args.batch_sizes:
        print(f"Benchmarking batch size {batch_size}...")
        batch_runs = []

        try:
            for warmup_index in range(args.warmup_runs):
                run_generation_pass(
                    model=model,
                    sentences=sentences,
                    batch_size=batch_size,
                    language=args.language,
                    base_prompt=base_prompt,
                )
                print(
                    f"  Warmup {warmup_index + 1}/{args.warmup_runs} complete for batch size {batch_size}"
                )

            for run_index in range(args.benchmark_runs):
                run_metrics = benchmark_batch_size(
                    model=model,
                    sentences=sentences,
                    batch_size=batch_size,
                    language=args.language,
                    base_prompt=base_prompt,
                )
                batch_runs.append(run_metrics)
                print(
                    f"  Timed run {run_index + 1}/{args.benchmark_runs}: "
                    f"{run_metrics['audio_seconds_per_second']:.2f} audio-s/s, "
                    f"{format_gib(run_metrics['peak_allocated_gib'])} peak alloc"
                )

        except RuntimeError as exc:
            if is_cuda_oom_error(exc):
                clear_cuda_after_failure()
                print(f"  Batch size {batch_size} OOM'd on the GPU. Skipping.")
                print()
                continue
            raise

        averaged = average_run_metrics(batch_runs)
        successful_results.append(averaged)
        print(format_summary_line(averaged, args.benchmark_runs))
        print()

    if not successful_results:
        print("No successful benchmark runs completed.")
        return

    best_result = max(
        successful_results,
        key=lambda metrics: metrics["audio_seconds_per_second"],
    )

    print("Best throughput result")
    print(
        f"  Batch size {best_result['batch_size']} achieved "
        f"{best_result['audio_seconds_per_second']:.2f} audio-seconds/sec "
        f"with {format_gib(best_result['peak_allocated_gib'])} peak allocated VRAM "
        f"and {format_gib(best_result['peak_reserved_gib'])} peak reserved VRAM."
    )


if __name__ == "__main__":
    main()
