import torch
import time

# Setup
batch_size = 256
height, width = 256, 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generate random mask coordinates
mask_coords = torch.randint(0, 200, (batch_size, 4), device=device)

# Method 1: Python for loop
def python_loop():
    tensor = torch.ones(batch_size, height, width, device=device)
    for i in range(batch_size):
        x, y, h, w = mask_coords[i]
        tensor[i, y:y+h, x:x+w] = 0
    return tensor

# Method 2: Vectorized
@torch.compile
def vectorized():
    tensor = torch.ones(batch_size, height, width, device=device)
    y_grid = torch.arange(height, device=device).view(1, -1, 1)
    x_grid = torch.arange(width, device=device).view(1, 1, -1)
    
    x_start = mask_coords[:, 0].view(-1, 1, 1)
    y_start = mask_coords[:, 1].view(-1, 1, 1)
    x_end = x_start + mask_coords[:, 2].view(-1, 1, 1)
    y_end = y_start + mask_coords[:, 3].view(-1, 1, 1)
    
    mask = (x_grid >= x_start) & (x_grid < x_end) & (y_grid >= y_start) & (y_grid < y_end)
    tensor[mask] = 0
    return tensor

# Method 3: Compiled loop (PyTorch 2.0+)
@torch.compile
def compiled_loop():
    tensor = torch.ones(batch_size, height, width, device=device)
    for i in range(batch_size):
        x, y, h, w = mask_coords[i]
        tensor[i, y:y+h, x:x+w] = 0
    return tensor


# Benchmark function
def benchmark(func, name, warmup=10, iterations=100):
    # Warmup
    for _ in range(warmup):
        func()
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Time it
    start = time.time()
    for _ in range(iterations):
        func()
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    end = time.time()
    avg_time = (end - start) / iterations * 1000  # Convert to ms
    print(f"{name}: {avg_time:.2f} ms")
    return avg_time

# Run benchmarks
print(f"Device: {device}")
print(f"Batch size: {batch_size}, Image size: {height}x{width}")
print("-" * 50)

loop_time = benchmark(python_loop, "Python for loop")
vec_time = benchmark(vectorized, "Vectorized")

# Only run compiled version if PyTorch 2.0+
if hasattr(torch, 'compile'):
    comp_time = benchmark(compiled_loop, "Compiled loop")
    print(f"\nSpeedup vectorized vs compiled: {comp_time/vec_time:.1f}x")

print(f"\nSpeedup vectorized vs loop: {loop_time/vec_time:.1f}x faster")