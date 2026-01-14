# Core LoRA Parameters
model: The model object returned by from_pretrained.

r (Rank): The most important hyperparameter. It determines the "width" of the adapter matrices. A higher r (e.g., 32, 64) allows the model to learn more complex patterns but uses more VRAM and is slower. Unsloth suggests 16 is a great middle ground.

target_modules: A list of the specific parts of the Transformer architecture to "attach" adapters to.

q, k, v, o_proj: Attention layers (helps with instruction following).

gate, up, down_proj: MLP/Feed-forward layers (helps with factual knowledge).

Unsloth optimizes all 7 for maximum accuracy.

lora_alpha: The scaling factor for the adapters. A common rule of thumb is to set this equal to r or 2×r. It controls how much "influence" the new training has over the original model weights.

lora_dropout: The probability of dropping units during training to prevent overfitting. In Unsloth, this is highly recommended to be 0.0 for maximum speed and efficiency.

bias: Usually set to "none". Setting it to "all" or "lora_only" would allow the bias terms to be trained, which is rarely necessary for standard fine-tuning.

# Unsloth & Advanced Logic
use_gradient_checkpointing: Set to "unsloth". This is the "secret sauce" that allows Unsloth to re-calculate activations during the backward pass, drastically reducing memory usage compared to the standard Hugging Face version.

use_rslora (Rank-Stabilized LoRA): If True, it uses a different scaling method (alpha/ r) which can lead to more stable training if you are using very high ranks (like r=256).

random_state: The seed (3407) for initializing the adapter weights to ensure your results can be reproduced.

init_lora_weights: If True, initializes the adapters (usually with Gaussian noise or zeros). If set to False, the adapters start with random junk—only use False if you are loading existing weights.

loftq_config: Used for LoftQ, a technique that quantizes the backbone and initializes the LoRA weights simultaneously to reduce the "quantization error."

modules_to_save: If you want to train layers other than the adapters (like the head or the embedding layer), you list them here. This is common when doing "Chat" fine-tuning where you add new special tokens.

# Technical / Internal
ensure_weight_tying: Some models (like Gemma) share weights between the input embeddings and output head. This ensures they stay synchronized during training.

temporary_location: Where Unsloth stores temporary buffers during the patching process to keep the GPU clean.

qat_scheme: Quantization-Aware Training scheme. Advanced usage for preparing models for specific hardware deployment (like 4-bit edge devices).