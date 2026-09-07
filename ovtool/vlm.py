"""Multimodal (vision-language) generation via openvino_genai.VLMPipeline."""
from __future__ import annotations

import argparse

import numpy as np
import openvino as ov
import openvino_tokenizers  # noqa: F401  (registers custom-op extension)

from .llm import build_generation_config, compile_options, make_streamer


def load_images(paths: list[str]) -> list[ov.Tensor]:
    from PIL import Image
    tensors = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        chw = np.asarray(img).transpose(2, 0, 1)  # HWC -> CHW, uint8
        tensors.append(ov.Tensor(chw[None, ...]))  # add batch dim, per official VLM sample
    return tensors


def run_vlm(args: argparse.Namespace) -> None:
    import openvino_genai as ovgenai

    from .devices import resolve_device
    device = resolve_device(args.device)
    opts = compile_options(args)
    print(f"[ovtool] loading VLM {args.model} on {device} ...")
    kwargs = dict(device=device)
    if opts:
        kwargs["config"] = opts
    pipe = ovgenai.VLMPipeline(args.model, **kwargs)
    cfg = build_generation_config(args)
    images = load_images(args.image) if args.image else None

    gen_kwargs = dict(generation_config=cfg, streamer=make_streamer(True))
    if images is not None:
        gen_kwargs["images"] = images

    print("Assistant > ", end="", flush=True)
    result = pipe.generate(args.prompt, **gen_kwargs)
    if result and args.stats:
        stats = getattr(result, "perf_metrics", None)
        if stats is not None:
            print(f"\n[stats] TTFT {stats.get_ttft().mean:.1f}ms | "
                  f"throughput {stats.get_throughput().mean:.1f} tok/s")


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("vlm", help="Vision-language (image+text) generation",
                       description="Multimodal generation: ask questions about images "
                                   "with a converted VLM (LLaVA / Qwen-VL / MiniCPM-V / InternVL ...).")
    p.add_argument("-m", "--model", required=True, help="Model directory (OpenVINO IR from 'convert vlm')")
    p.add_argument("-d", "--device", default="CPU", help="Inference device (default CPU)")
    p.add_argument("prompt", help="Question / instruction text")
    p.add_argument("-i", "--image", action="append", help="Image path (repeatable)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument("--rng-seed", type=int, default=None)
    p.add_argument("--stop-tokens", nargs="*", default=None)
    p.add_argument("--opt", action="append", metavar="KEY=VALUE", help="Runtime option KEY=VALUE (repeatable)")
    p.add_argument("--stats", action="store_true", help="Print performance stats")
    p.set_defaults(func=run_vlm)
