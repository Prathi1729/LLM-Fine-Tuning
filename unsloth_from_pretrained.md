# Core Configuration
model_name: The Hugging Face repo ID (e.g., "unsloth/llama-3-8b-bnb-4bit"). Unsloth works best with their pre-quantized models, but can also convert standard FP16 models on the fly.

max_seq_length: The maximum context window for training. Unsloth automatically handles RoPE Scaling if you set this higher than the model’s native limit.

dtype: The precision of the weights. Usually None (auto-detects). It will typically use torch.bfloat16 on Ampere (A100/H100/RTX 30/40 series) or torch.float16 on older GPUs.

random_state: The seed for reproducibility. (3407 is the "magic number" from the Unsloth team that they jokingly claim yields better accuracy).

# Quantization & Precision
These parameters determine how much GPU VRAM the model will consume.

load_in_4bit: Enables 4-bit QLoRA. This is the standard for Unsloth, allowing an 8B model to fit in under 6GB of VRAM.

load_in_8bit: Uses 8-bit quantization. Slightly more accurate than 4-bit, but uses more memory.

load_in_16bit: Loads the model in half-precision (FP16/BF16). No quantization; requires significantly more VRAM.

load_in_fp8: A newer feature for H100/L40S GPUs. It uses 8-bit floating point, which is faster and more accurate than 4-bit integer quantization.

float32_mixed_precision: Forces the model to use FP32 for certain calculations. Rarely used unless you encounter extreme stability issues.

# Model Architecture & Patches
rope_scaling: Used to extend context. If you want a 4k model to handle 16k tokens, you provide a dictionary like {"type": "linear", "factor": 4}.

use_gradient_checkpointing: Set to "unsloth" (default). This uses Unsloth's optimized version of gradient checkpointing which re-calculates activations to save massive amounts of memory.

fix_tokenizer: Automatically fixes common issues with Llama/Gemma tokenizers (like missing pad tokens or incorrect BOS/EOS settings).

resize_model_vocab: If you add new tokens (like special chat tags), this resizes the embedding layers to accommodate them.

unsloth_tiled_mlp: An experimental optimization for Llama-3 that speeds up the Feed-Forward Network (MLP) layers by tiling computations.

# Memory & Performance Tuning
device_map: Determines which GPU(s) the model is loaded onto. "sequential" or "auto" are standard.

offload_embedding: If you are extremely low on VRAM, this moves the embedding layers to the System RAM (CPU) instead of the GPU.

max_lora_rank: Sets a ceiling on the LoRA Rank (r). Usually, r=16 or r=32 is plenty.

gpu_memory_utilization: Used when fast_inference is True to tell vLLM how much of the GPU it is allowed to "reserve."

# Inference & Deployment
fast_inference: If True, Unsloth will attempt to use vLLM for the generation phase. This makes the model respond much faster but requires extra setup.

float8_kv_cache: Compresses the Key-Value (KV) cache to 8-bit. This allows you to handle massive context lengths (like 128k) without running out of memory during generation.

# Administrative / Backend
token: Your Hugging Face Hub token (required for gated models like Llama 3 or Mistral).

trust_remote_code: Must be True for models with custom architectures not yet in the transformers library.

full_finetuning: If True, it ignores LoRA and attempts to update every single parameter. (Requires massive VRAM, e.g., 80GB+ for an 8B model).

qat_scheme: Quantization-Aware Training. Used for advanced users trying to prepare a model for deployment on specific hardware (like mobile chips).

disable_log_stats: Silences the Unsloth "heartbeat" and version logs in the console.