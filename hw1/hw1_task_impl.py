import torch


# ============================================================================
# Part 1: Implement PyTorch Functions
# ============================================================================
#
# TASK 1a: Implement an operation with the lowest arithmetic intensity.
# Use an op that performs essentially memory traffic with ~0 useful FLOPs
# per element.


def lowest_ai_fn(x: torch.Tensor) -> torch.Tensor:
    """Lowest arithmetic intensity baseline (0 FLOP/Byte)."""
    return x.clone()


# TASK 1b: Implement a function with configurable arithmetic intensity.
# Build an element-wise compute operation where work increases with `num_ops`.
# Design it so fused arithmetic intensity grows roughly linearly with `num_ops`,
# while each element is still read/written once at the kernel boundary.
# Return either the eager function or a compiled version depending on the
# `compiled` flag so we can compare both on the roofline plot.
#
# Use an accumulator variable and implement fused multiply-add (FMA) style work
# explicitly, e.g. `acc = acc * x + x`, so each loop iteration contributes
# about 2 FLOPs per element in a realistic GPU-friendly pattern. We prefer this
# pattern here mainly because it gives clean FLOP accounting and resembles the
# kind of floating-point work GPUs are designed to do; Avoid patterns like repeated
# doubling (`x = x + x`), since long self-dependent pointwise chains can trigger
# very poor Inductor compile-time behavior and are also less useful for this
# roofline exercise.


def make_compute_fn(num_ops: int, compiled: bool = True):
    """Return an eager or compiled function whose work scales with num_ops."""
    def fn(x: torch.Tensor) -> torch.Tensor:
        acc = x
        for _ in range(num_ops):
            acc = acc * x + x
        return acc

    if compiled:
        try:
            return torch.compile(fn)
        except (RuntimeError, AttributeError, Exception):
            # Fallback for Python 3.14+ or other environments where torch.compile fails
            return fn
    return fn


# ============================================================================
# Part 2: Benchmarking
# ============================================================================
#
# TASK 2: Complete the benchmark function using CUDA events.
# CUDA events measure GPU time precisely (not CPU wall time), which avoids
# including kernel launch overhead or CPU-GPU synchronization delays.


def benchmark_fn(fn, *args, warmup=25, rep=100) -> float:
    """Benchmark a function, falling back to CPU time if CUDA is not available."""
    import time

    # Warmup
    for _ in range(warmup):
        fn(*args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep)]

        for i in range(rep):
            start_events[i].record()
            fn(*args)
            end_events[i].record()

        torch.cuda.synchronize()
        times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    else:
        # Fallback to CPU timing
        times = []
        for _ in range(rep):
            t0 = time.perf_counter()
            fn(*args)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # to ms

    return float(torch.tensor(times).median().item())


# TASK 3: Compute element-wise operation metrics from measured runtime.
# Count every arithmetic operation performed inside the loop (careful: each
# `acc = acc * x + x` iteration does more than one FLOP per element).
#
# Use different byte-traffic models for the two variants:
#   - compiled: assume the operation is fused, so each element is read once and
#     written once at the kernel boundary
#   - eager: estimate the traffic from the separate multiply and add operations
#     launched by PyTorch in each loop iteration, including intermediate tensors
#
# Return a tuple with:
#   - total_flops
#   - arithmetic_intensity  (FLOP / Byte)
#   - achieved_flops        (FLOP / s)


def compute_elementwise_metrics(num_elements, num_ops, bytes_per_element, ms, variant):
    # Each acc = acc * x + x iteration does 1 mul and 1 add = 2 FLOPs per element.
    # Total ops = num_ops iterations.
    flops_per_element = 2 * num_ops
    total_flops = num_elements * flops_per_element
    
    # Bytes moved:
    # Compiled/Fused: Read x once, write acc once = 2 * num_elements * bytes_per_element
    if variant == "compiled":
        total_bytes = 2 * num_elements * bytes_per_element
    else:
        # Eager: Separate multiply and add operations.
        # acc = acc * x + x
        # 1. tmp = acc * x (reads acc, reads x, writes tmp) -> 3 elements
        # 2. acc = tmp + x (reads tmp, reads x, writes acc) -> 3 elements
        # Total per iteration = 6 elements * bytes_per_element
        total_bytes = num_ops * 6 * num_elements * bytes_per_element
        
        # Special case: num_ops = 0? Not really expected here but for completeness
        if num_ops == 0:
            total_bytes = 2 * num_elements * bytes_per_element

    ai = total_flops / total_bytes if total_bytes > 0 else 0
    achieved_flops = total_flops / (ms * 1e-3) if ms > 0 else 0
    
    return total_flops, ai, achieved_flops


# ============================================================================
# Part 3: Short Writeup
# ============================================================================
# Answer these after you generate `results/roofline.png` and inspect the points.
#
# Q1. Look at the compiled element-wise operations from `1 ops` through `64 ops`.
# Why does performance rise as arithmetic intensity increases even though the
# measured runtime changes only a little?
#
# A1: In this region (1 to 64 ops), the operation is strictly memory-bound. The 
# measured runtime remains nearly constant at ~0.18 ms because it is limited 
# by the GPU's HBM bandwidth (achieving ~2.96 TB/s in our data). Since we are 
# moving the same amount of data (reading 'x' and writing 'acc' once) while 
# increasing the number of FLOPs performed on that data, the achieved TFLOP/s 
# rises linearly with arithmetic intensity while the latency stays flat.
#
# Q2. In one sample run, `matmul 1024x1024` achieved lower FLOP/s than the
# `128 ops` compiled element-wise operation. Give one or two reasons why that can
# happen on a large GPU like an H100.
#
# A2: Our data shows the 128-ops element-wise operation reaching ~60.3 TFLOPS, 
# while the 1024x1024 matmul only hits ~36.9 TFLOPS. This happens because: 
# 1) Low Occupancy: A 1024x1024 matrix multiplication (2^20 elements) 
# does not provide enough thread blocks to fully saturate the massive number 
# of SMs on an H100. 2) Specialized vs. General: The element-wise kernel 
# has extremely high register reuse and simple, perfectly coalesced memory 
# access patterns compared to the shared memory and tiling overheads 
# inherent in GEMM.
#
# Q3. Between `64 ops` and `128 ops`, runtime increases more noticeably than it
# did for smaller operations. What does that suggest about what resource is
# becoming the bottleneck?
#
# A3: The runtime jumps from ~0.18 ms at 64 ops to ~0.28 ms at 128 ops. This 
# indicates we have crossed the "ridge point" of the roofline and the 
# kernel has transitioned from being memory-bound to compute-bound. 
# At this point, the throughput of the floating-point units (ALUs) 
# becomes the bottleneck, so doubling the work (from 64 to 128 ops) now 
# results in a significant increase in execution time.
#
# Q4. Why do the eager `ops-K` points look so different from the compiled ones?
#
# A4: Eager mode fails to fuse the operations, launching separate kernels for 
# each addition and multiplication in the loop. This results in: 
# 1) Massive Memory Traffic: Intermediate results are written to and read 
# from VRAM, keeping arithmetic intensity extremely low (~0.083 FLOP/Byte 
# in our results). 2) Launch Overhead: The cumulative overhead of 
# launching hundreds of separate kernels dominates the runtime, as seen 
# in the high latency (~67 ms for 128 eager ops vs 0.28 ms for compiled).
