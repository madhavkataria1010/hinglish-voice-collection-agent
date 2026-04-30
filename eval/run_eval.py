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

from .corpus import load_corpus  # noqa: E402
from .metrics import (  # noqa: E402
    aggregate,
    render_report,
    run_baseline_pipeline,
    run_our_pipeline,
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
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    logger.info(f"Loaded {len(corpus)} clips from {args.corpus}")

    ours_metrics = baseline_metrics = None

    if args.pipeline in ("ours", "both"):
        logger.info("Running OUR pipeline (faster-whisper + Sarvam + normalizer)")
        ours_results = await run_our_pipeline(corpus)
        ours_metrics = aggregate("ours", ours_results)

    if args.pipeline in ("baseline", "both"):
        logger.info("Running BASELINE pipeline (Deepgram + OpenAI TTS)")
        baseline_results = await run_baseline_pipeline(corpus)
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
