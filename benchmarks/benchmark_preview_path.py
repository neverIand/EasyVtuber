"""Measure the CPU-only staging and shared-memory preview path."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.utils.preview_ipc import PreviewFrameFormatter, PreviewSharedBuffer


CASES = (
    (512, 3, 'BGR'),
    (1024, 3, 'BGR'),
    (1024, 4, 'RGBA'),
)


def percentile(samples, value):
    return float(np.percentile(samples, value))


def summarize(samples):
    return (
        f'median={statistics.median(samples):.3f} ms, '
        f'p95={percentile(samples, 95):.3f} ms, '
        f'max={max(samples):.3f} ms'
    )


def measure(size, channels, source_format, iterations, warmup):
    source = np.random.default_rng(size + channels).integers(
        0,
        256,
        size=(size, size, channels),
        dtype=np.uint8,
    )
    staging = np.empty_like(source)
    formatter = PreviewFrameFormatter()
    transport = PreviewSharedBuffer.create()

    copy_samples = []
    publish_samples = []
    total_samples = []
    try:
        for index in range(warmup + iterations):
            started = time.perf_counter()
            np.copyto(staging, source)
            copied = time.perf_counter()
            transport.publish_rgba(formatter.format(staging, source_format))
            finished = time.perf_counter()

            if index >= warmup:
                copy_samples.append((copied - started) * 1000)
                publish_samples.append((finished - copied) * 1000)
                total_samples.append((finished - started) * 1000)
    finally:
        transport.close()

    print(f'{size}x{size} {source_format}')
    print(f'  output-slot copy: {summarize(copy_samples)}')
    print(f'  post-release format + publish: {summarize(publish_samples)}')
    print(f'  end-to-end preview path: {summarize(total_samples)}')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark the launcher preview path without starting a GPU model.',
    )
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--warmup', type=int, default=20)
    return parser.parse_args()


def main():
    cli = parse_args()
    if cli.iterations <= 0 or cli.warmup < 0:
        raise SystemExit('--iterations must be positive and --warmup non-negative')
    for case in CASES:
        measure(*case, iterations=cli.iterations, warmup=cli.warmup)


if __name__ == '__main__':
    main()
