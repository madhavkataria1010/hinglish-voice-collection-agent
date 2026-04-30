"""
Single command: produce eval/report.md comparing baseline vs. our pipeline.

    python -m eval.run_eval                   # both pipelines
    python -m eval.run_eval --pipeline ours
    python -m eval.run_eval --pipeline baseline

If `eval/test_corpus/manifest.jsonl` is missing, prints a clear error.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

import json

from .corpus import load_corpus  # noqa: E402
from .metrics import (  # noqa: E402
    PerClipResult,
    aggregate,
    render_report,
    run_baseline_pipeline,
    run_our_pipeline,
)


def _dump_raw(results: list[PerClipResult], path: Path) -> None:
    """Per-clip raw data, JSONL, so the report's claims are auditable."""
    with path.open("w") as f:
        for r in results:
            f.write(
                json.dumps(
                    {
                        "audio": r.item.audio_path.name,
                        "category": r.item.category,
                        "gold_transcript": r.item.transcript,
                        "gold_language": r.item.language,
                        "gold_amounts": r.item.amounts_inr,
                        "stt_text": r.stt_text,
                        "stt_language": r.stt_language,
                        "stt_latency_ms": round(r.stt_latency_s * 1000.0, 1),
                        "pipeline_language": r.pipeline_language,
                        "extracted_amounts": r.extracted_amounts,
                        "tts_ttfb_ms": (
                            round(r.tts_ttfb_s * 1000.0, 1)
                            if r.tts_ttfb_s is not None
                            else None
                        ),
                        "perceived_latency_ms": (
                            round(r.perceived_latency_ms, 1)
                            if r.perceived_latency_ms is not None
                            else None
                        ),
                    }
                )
                + "\n"
            )


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pipeline",
        choices=("ours", "baseline", "both"),
        default="both",
    )
    parser.add_argument("--corpus", default="eval/test_corpus")
    parser.add_argument("--out", default="eval/report.md")
    parser.add_argument(
        "--raw-dir",
        default="eval/results",
        help="Where to write per-clip JSONL raw data.",
    )
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    logger.info(f"Loaded {len(corpus)} clips from {args.corpus}")

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    ours_metrics = baseline_metrics = None

    if args.pipeline in ("ours", "both"):
        logger.info("Running OUR pipeline (faster-whisper + Sarvam + normalizer)")
        ours_results = await run_our_pipeline(corpus)
        _dump_raw(ours_results, raw_dir / "ours.jsonl")
        ours_metrics = aggregate("ours", ours_results)

    if args.pipeline in ("baseline", "both"):
        logger.info("Running BASELINE pipeline (Deepgram + OpenAI TTS)")
        baseline_results = await run_baseline_pipeline(corpus)
        _dump_raw(baseline_results, raw_dir / "baseline.jsonl")
        baseline_metrics = aggregate("baseline", baseline_results)

    if ours_metrics and baseline_metrics:
        report = render_report(ours_metrics, baseline_metrics)
    else:
        chosen = ours_metrics or baseline_metrics
        assert chosen is not None
        report = (
            f"# Evaluation report (single pipeline: {chosen.pipeline_name})\n\n"
            f"- n_clips: {chosen.n_clips}\n"
            f"- perceived_latency_p50_ms: {chosen.perceived_latency_p50_ms:.0f}\n"
            f"- perceived_latency_p95_ms: {chosen.perceived_latency_p95_ms:.0f}\n"
            f"- language_accuracy: {chosen.language_accuracy:.1%}\n"
            f"- numeric_preservation: {chosen.numeric_preservation:.1%}\n"
            f"- false_switch_rate: {chosen.false_switch_rate:.1%}\n"
        )
    Path(args.out).write_text(report)
    logger.info(f"Wrote {args.out}")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
