Here’s a crisp, practical **short notes** version of *How to use your Mac GPU with PyTorch*, perfect for quick reference.

---

## **Short Notes: Using Mac GPU (MPS) with PyTorch**

### **1. CUDA will never work on macOS**
- CUDA is **NVIDIA‑only**  
- macOS does **not support NVIDIA GPUs or CUDA drivers**  
- Therefore:  
  ```python
  torch.cuda.is_available()  # always False on Mac
  ```

---

### **2. Mac uses MPS (Metal Performance Shaders) instead**
PyTorch supports Apple GPUs through **MPS**, not CUDA.

Check availability:
```python
import torch
print(torch.backends.mps.is_available())  # True if GPU usable
print(torch.backends.mps.is_built())      # True if PyTorch built with MPS
```

---

### **3. Install PyTorch with MPS support**
Works on macOS 12.3+:

```bash
pip install torch torchvision torchaudio
```

No special flags needed.

---

### **4. Use MPS device in your code**
```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
inputs = inputs.to(device)
```

---

### **5. Example: Run a tensor on MPS**
```python
import torch

device = torch.device("mps")
x = torch.randn(3,3).to(device)
print(x)
```

---

### **6. Benchmark CPU vs MPS**
Warm-up + synchronize for accurate timing:

```python
import torch, time

size = 5000
a = torch.randn(size, size)
b = torch.randn(size, size)

# CPU
start = time.time()
_ = a @ b
print("CPU:", time.time() - start)

# MPS
if torch.backends.mps.is_available():
    a_mps = a.to("mps")
    b_mps = b.to("mps")

    _ = a_mps @ b_mps  # warm-up
    torch.mps.synchronize()

    start = time.time()
    _ = a_mps @ b_mps
    torch.mps.synchronize()
    print("MPS:", time.time() - start)
```

---

### **7. When MPS is slower**
- First warm-up call  
- Some ops not fully optimized  
- Large models may hit unified memory limits  

---

### **8. When you need CUDA**
Use:
- Linux + NVIDIA  
- Windows + NVIDIA  
- Cloud GPUs (Colab, RunPod, AWS, Lambda Labs)

---

If you want, I can turn this into a **one‑page cheat sheet**, **PDF‑ready notes**, or **step‑by‑step guide**.