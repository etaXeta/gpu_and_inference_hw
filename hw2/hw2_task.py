import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    # HW1 taught us that eager mode and frequent syncs (like .item()) are slow.
    # We use KV caching to avoid quadratic re-computation and torch.compile for fusion.
    generated_tokens = []
    
    # Initial forward pass to populate KV cache
    outputs = model(input_ids=input_ids, use_cache=True)
    past_key_values = outputs.past_key_values
    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
    
    # Pre-allocate output tensor on GPU to avoid .item() inside the loop
    # We still need to return a list of Python ints for compatibility with utils.py
    # but we can collect them all at once at the very end.
    all_token_ids = torch.zeros(n_steps, dtype=torch.long, device=input_ids.device)
    all_token_ids[0] = next_token_id
    
    curr_input_ids = next_token_id.unsqueeze(0)
    
    for i in range(1, n_steps):
        outputs = model(input_ids=curr_input_ids, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        all_token_ids[i] = next_token_id
        curr_input_ids = next_token_id.unsqueeze(0)
    
    return all_token_ids.tolist()


def profile(loop_fn, model, input_ids, trace_name: str):
    # HW2: wrap loop_fn(model, input_ids, PROFILE_STEPS) with torch.profiler,
    # print the summary table, and export a Chrome trace to RESULTS_DIR / trace_name
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)

    prof.export_chrome_trace(str(RESULTS_DIR / trace_name))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def generate_optimized(optimized_trace_name: str) -> float:
    # 1. Use BF16 to reduce memory bandwidth requirements (HW1: Memory-bound region benefit)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = build_model(dtype)
    
    # 2. Use torch.compile to fuse kernels and reduce launch overhead (HW1: Compiled vs Eager)
    # mode="reduce-overhead" is great for small models like this tiny Llama.
    model = torch.compile(model, mode="reduce-overhead")
    
    input_ids = get_input_ids()

    # Warmup for torch.compile
    print("Warming up optimized model...")
    optimized_loop(model, input_ids, 5)

    profile(optimized_loop, model, input_ids, optimized_trace_name)
    optimized_elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    return optimized_elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
# 1. KV Caching: Switched from quadratic O(N^2) re-computation to O(N) by passing `past_key_values`. 
#    This drastically reduces memory traffic and compute per step.
# 2. BF16 Precision: Used `torch.bfloat16` instead of `float32`. 
#    This halves memory bandwidth requirements, which is critical as small-batch inference is memory-bound.
# 3. Eliminated .item() Syncs: Kept token IDs on GPU during the loop and used `.tolist()` at the very end. 
#    This prevents frequent CPU-GPU synchronization bubbles.
# 4. torch.compile: Used `torch.compile(mode="reduce-overhead")` to fuse kernels and eliminate Python/framework overhead.
#
# Biggest impact and why:
# KV Caching had the biggest impact. Without it, the model re-processes the entire prompt and all 
# generated tokens at every single step, leading to quadratic growth in work. 
# Even with all other optimizations, O(N^2) would eventually dominate latency.
