import torch
import time

# Choose matrix size
size = 5000

# Create CPU tensors
a = torch.randn(size, size)
b = torch.randn(size, size)

# ---------------------------
# CPU Benchmark
# ---------------------------
start = time.time()
c_cpu = a @ b
cpu_time = time.time() - start
print(f"CPU time: {cpu_time:.4f}s")

# ---------------------------
# MPS Benchmark
# ---------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")

    # Move tensors to MPS
    a_mps = a.to(device)
    b_mps = b.to(device)

    # Warm-up (VERY important for MPS)
    _ = a_mps @ b_mps
    torch.mps.synchronize()

    # Actual timed run
    start = time.time()
    c_mps = a_mps @ b_mps
    torch.mps.synchronize()
    mps_time = time.time() - start

    print(f"MPS time: {mps_time:.4f}s")
else:
    print("MPS backend not available.")
