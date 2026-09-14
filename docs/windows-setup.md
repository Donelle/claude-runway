# LM Studio GPU Setup Guide: NVIDIA RTX A2000 + Intel Optimus (Windows)

Step-by-step setup for running LM Studio with CUDA hardware acceleration on laptops with dual graphics (Intel UHD integrated + a dedicated NVIDIA GPU). This targets the specific hardware combination most many corporate-issued laptops ship with, but the same steps apply to any Windows laptop with Intel/NVIDIA Optimus hybrid graphics.

This is relevant to this repo's local-compress piece (`compress_mcp_server.py`), which depends on LM Studio running locally with a model loaded — see the README's Prerequisites section.

---

## Hardware Specifications & Diagnosis

* **CPU:** Intel Core i7-11800H
* **Integrated GPU (iGPU):** Intel UHD Graphics (PCI Bus 0)
* **Discrete GPU (dGPU):** NVIDIA RTX A2000 Laptop GPU (4GB VRAM, PCI Bus 1)
* **System RAM:** 64 GB

### The Problem

1. **CUDA failure:** Windows Optimus puts the NVIDIA RTX A2000 into sleep mode (`D3Cold`) to save battery when desktop apps launch. CUDA fails to detect the card on startup and reports `0 devices`.
2. **Vulkan fallback issue:** Switching to Vulkan allocates model weight memory to the RTX A2000, but routes matrix math compute to the Intel iGPU (GPU 0), resulting in slow inference and 100% iGPU usage.

---

## Step-by-Step Solution

### Step 1: Force NVIDIA Control Panel to Use Dedicated GPU

1. Right-click your desktop and open **NVIDIA Control Panel**.
2. Under **3D Settings**, click **Manage 3D Settings**.
3. Select the **Program Settings** tab.
4. Click **Add** and select **LM Studio** (`LM Studio.exe`).
   * *If not in the list, browse to:* `%LOCALAPPDATA%\Programs\LM-Studio\LM Studio.exe`
5. Under **Select the preferred graphics processor for this program**, choose:
   > **High-performance NVIDIA processor**
6. *(Optional)* Add `llama-server.exe` from the LM Studio subfolders and set it to **High-performance NVIDIA processor** as well.
7. Click **Apply** at the bottom right.

---

### Step 2: Set Windows Display Graphics to High Performance

1. Open **Windows Settings** (`Win + I`).
2. Go to **System** -> **Display** -> **Graphics** (or search *"Graphics Settings"* in the Start Menu).
3. Click **Browse** under *Custom options for apps*.
4. Select `LM Studio.exe` (`AppData\Local\Programs\LM-Studio\LM Studio.exe`).
5. Click **Options**, select **High Performance (NVIDIA RTX A2000)**, and click **Save**.

---

### Step 3: Configure LM Studio for CUDA

1. Restart your PC (or terminate all `LM Studio`, `lmstudio-backend`, and `node` tasks in Task Manager).
2. Open **LM Studio**.
3. In your model's **Load Parameters** (right sidebar or My Models gear icon):
   * Set **GPU Backend** to **CUDA** (or Auto).
   * Set **GPU Offload** to **12 layers** (see Optimal Load Parameters below for why not Max).
4. LM Studio will now detect and initialize the **NVIDIA RTX A2000**.

---

## Optimal Load Parameters (For 4GB VRAM)

Because the RTX A2000 has **4GB VRAM**, optimize load parameters to prevent Out-Of-Memory (`cudaMalloc`) errors:

| Parameter | Recommended Setting | Reason |
| :--- | :--- | :--- |
| **Model Size** | 3B - 4B models (e.g. `gemma-3-4b`, `phi-3-mini`, `qwen-2.5-3b`) | Q4_K_M quantization consumes ~2.5 GB VRAM. |
| **Context Length (`n_ctx`)** | `1024` or `2048` tokens | Preserves ~1.5 GB VRAM for the KV cache buffer. |
| **Flash Attention** | Enabled | Reduces KV cache memory footprint on CUDA Tensor Cores. |
| **GPU Offload** | `12` layers | Offloading past ~20 layers is too large to fit in 4GB VRAM alongside the context memory buffer, once CUDA is actually working. |

---

## Verification & Diagnostics

To verify that your RTX A2000 is doing the work:

1. Open **Task Manager** (`Ctrl + Shift + Esc`).
2. Go to the **Performance** tab -> select your **NVIDIA RTX A2000** (typically **GPU 1**).
3. Right-click the performance graph -> **Change graph to** -> **CUDA** or **Compute_0**.
4. Send a prompt in LM Studio:
   * **VRAM usage:** Should show ~2.5 GB - 3.5 GB allocated.
   * **CUDA graph:** Should spike during prompt processing and generation.
   * **Intel iGPU (GPU 0):** Should stay near 0%.

---

## Troubleshooting: Processing Occasionally Shifts to the Integrated GPU

If GPU 0 (the Intel iGPU) intermittently spikes mid-conversation even after the setup above, it's usually Windows WDDM VRAM spillover or NVIDIA power management — not a misconfiguration from Step 1-3.

### Cause 1: VRAM Spillover (Windows "Shared GPU Memory")

The RTX A2000 has 4GB of VRAM.

* As a chat conversation gets longer, the KV cache grows.
* Once total memory hits 3.8 GB - 4.0 GB, Windows automatically pages the extra memory out into Shared GPU Memory (system RAM).
* Since system RAM is managed by the CPU and Intel iGPU, Windows routes that overflow processing through GPU 0 (Intel iGPU), causing a sudden slowdown and a GPU 0 spike.

### Cause 2: NVIDIA Power Management Throttling

Between generated tokens, if the RTX A2000 briefly drops into low-power mode, Windows Optimus hands background buffer transfers back to the Intel iGPU.

### Fix 1: Disable CUDA System Memory Fallback in NVIDIA Control Panel

Prevents Windows from silently spilling VRAM over to GPU 0/system RAM.

1. Open NVIDIA Control Panel.
2. Go to **Manage 3D Settings** -> **Program Settings** -> select LM Studio.
3. Scroll down to **CUDA - Sysmem Fallback Policy**.
4. Change it to **Prefer No Sysmem Fallback**.
5. Click **Apply**.

### Fix 2: Set Power Management to "Prefer Maximum Performance"

1. In NVIDIA Control Panel -> **Manage 3D Settings** -> **Program Settings** (LM Studio).
2. Scroll down to **Power Management Mode**.
3. Change it to **Prefer maximum performance**.
4. Click **Apply**.

### Fix 3: Leave a 500MB VRAM Safety Cushion in LM Studio

To guarantee the 4GB RTX A2000 never hits 100% capacity:

* If using a 4B model (e.g. 33 layers), set **GPU Offload** to `12` layers instead of offloading all of them, or set **Context Length** to `2048`.
* Leaving ~500MB of free VRAM prevents Windows WDDM from triggering the fallback to GPU 0.

### Fix 4: Restrict Vulkan to the Discrete GPU (`GGML_VK_VISIBLE_DEVICES`)

If LM Studio is using the Vulkan backend, its underlying llama.cpp engine can still see and route work to both GPUs even after Fixes 1-3, since NVIDIA Control Panel settings only affect NVIDIA's own driver behavior, not which devices Vulkan enumerates. `GGML_VK_VISIBLE_DEVICES` is a llama.cpp environment variable that restricts the Vulkan backend to a specific device index, which is more deterministic than relying on LM Studio's UI alone:

1. Confirm the device index for the RTX A2000 from LM Studio's hardware/GPU settings panel (or run `vulkaninfo` from a terminal and note the order GPUs are listed in).
2. Open **Windows Settings** -> **System** -> **About** -> **Advanced system settings** -> **Environment Variables**.
3. Add a new system (or user) variable named `GGML_VK_VISIBLE_DEVICES` with the RTX A2000's device index as the value (e.g. `0` if it's listed first).
4. Restart LM Studio so it picks up the new environment variable.
5. Re-check Task Manager per the Verification steps above to confirm GPU 0 (Intel iGPU) stays idle.
