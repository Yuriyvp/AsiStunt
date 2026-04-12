"""Benchmark OmniVoice optimizations: SDPA/Flash Attention + torch.compile.

Runs a short TTS synthesis with different configurations and reports timings.
Requires GPU and loaded OmniVoice model.

Usage: python tests/bench_tts_optimizations.py
"""
import gc
import time
import sys
import os

import torch
import numpy as np


def load_model(attn_implementation=None):
    """Load OmniVoice with optional attention implementation."""
    from omnivoice import OmniVoice

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    kwargs = {
        "device_map": "cuda",
        "dtype": torch.float16,
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", **kwargs)
    model.eval()
    return model


def load_voice_prompt(lang="en"):
    """Load a voice profile for testing."""
    from voice_assistant.core.voice_clone import load_voice_clone_prompt, get_profile_path
    path = get_profile_path(f"voice_{lang}")
    return load_voice_clone_prompt(str(path))


def bench_generate(model, prompt, label, num_step=32, runs=5):
    """Benchmark generation with given model and config."""
    from omnivoice.models.omnivoice import OmniVoiceGenerationConfig

    text = "This is a benchmark test to measure text to speech synthesis speed on different optimization settings."
    config = OmniVoiceGenerationConfig(num_step=num_step)

    # Warmup (2 runs)
    for _ in range(2):
        with torch.inference_mode():
            model.generate(
                text=text,
                voice_clone_prompt=prompt,
                language="en",
                speed=1.0,
                generation_config=config,
            )
    torch.cuda.synchronize()

    # Benchmark
    times = []
    audio_durations = []
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.monotonic()
        with torch.inference_mode():
            result = model.generate(
                text=text,
                voice_clone_prompt=prompt,
                language="en",
                speed=1.0,
                generation_config=config,
            )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - t0
        audio_dur = result[0].shape[-1] / 24000
        times.append(elapsed)
        audio_durations.append(audio_dur)

    avg_time = sum(times) / len(times)
    avg_audio = sum(audio_durations) / len(audio_durations)
    rtf = avg_time / avg_audio if avg_audio > 0 else 0
    min_time = min(times)

    print(f"  {label}:")
    print(f"    steps={num_step}, runs={runs}")
    print(f"    avg={avg_time:.3f}s  min={min_time:.3f}s  audio={avg_audio:.2f}s  RTF={rtf:.3f}")
    print(f"    times: {[f'{t:.3f}' for t in times]}")
    return avg_time, rtf


def try_compile(model, label):
    """Try torch.compile on the model's LLM forward pass."""
    print(f"\n  Attempting torch.compile for {label}...")
    try:
        model.llm = torch.compile(model.llm, mode="reduce-overhead")
        print(f"  torch.compile applied successfully")
        return True
    except Exception as e:
        print(f"  torch.compile FAILED: {e}")
        return False


def main():
    print("=" * 70)
    print("OmniVoice TTS Optimization Benchmark")
    print("=" * 70)

    # Check current attention config
    print(f"\nPyTorch {torch.__version__}, CUDA {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SDPA flash backend: {torch.backends.cuda.flash_sdp_enabled()}")
    print(f"SDPA mem_efficient: {torch.backends.cuda.mem_efficient_sdp_enabled()}")

    prompt = load_voice_prompt("en")
    results = {}

    # --- Test 1: Baseline (default attention) ---
    print("\n" + "-" * 70)
    print("TEST 1: Baseline (default attention, num_step=32)")
    print("-" * 70)
    model = load_model()

    # Check what attention the LLM is using
    llm_config = model.llm.config
    attn_impl = getattr(llm_config, "_attn_implementation", "unknown")
    attn_impl_internal = getattr(llm_config, "_attn_implementation_internal", "unknown")
    print(f"  LLM attention impl: {attn_impl} (internal: {attn_impl_internal})")

    results["baseline_32"] = bench_generate(model, prompt, "Baseline 32 steps", num_step=32)
    results["baseline_16"] = bench_generate(model, prompt, "Baseline 16 steps", num_step=16)

    # --- Test 2: num_step=12 ---
    print("\n" + "-" * 70)
    print("TEST 2: Fewer steps (num_step=12)")
    print("-" * 70)
    results["baseline_12"] = bench_generate(model, prompt, "Baseline 12 steps", num_step=12)

    # --- Test 3: torch.compile on LLM ---
    print("\n" + "-" * 70)
    print("TEST 3: torch.compile (reduce-overhead)")
    print("-" * 70)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    model = load_model()
    compiled = try_compile(model, "compile LLM")
    if compiled:
        results["compiled_32"] = bench_generate(model, prompt, "Compiled, 32 steps", num_step=32, runs=5)
        results["compiled_16"] = bench_generate(model, prompt, "Compiled, 16 steps", num_step=16, runs=5)
    else:
        results["compiled_32"] = None
        results["compiled_16"] = None

    # --- Test 4: torch.compile with max-autotune ---
    if not compiled:
        print("\n  Trying torch.compile mode='default' as fallback...")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = load_model()
        try:
            model.llm = torch.compile(model.llm, mode="default")
            print("  torch.compile (default) applied")
            results["compiled_default_16"] = bench_generate(model, prompt, "Compiled default, 16 steps", num_step=16, runs=5)
        except Exception as e:
            print(f"  torch.compile (default) FAILED: {e}")
            results["compiled_default_16"] = None

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<35} {'Avg Time':>10} {'RTF':>8}")
    print("-" * 55)
    for key, val in results.items():
        if val is not None:
            avg_time, rtf = val
            print(f"  {key:<33} {avg_time:>8.3f}s {rtf:>8.3f}")
        else:
            print(f"  {key:<33} {'FAILED':>10}")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nDone.")


if __name__ == "__main__":
    main()
