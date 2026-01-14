from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

max_seq_length = 4096 # Supports RoPE Scaling automatically
dtype = None # None for auto detection. Float16 for Tesla T4, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization


########  MODEL LOADER #################
"""
    Loads a pre-trained language model with Unsloth optimizations for faster training and reduced memory usage.

    Args:
        model_name (str): The Hugging Face model ID or local path. Unsloth-optimized models (e.g., "unsloth/llama-3-8b-bnb-4bit") are recommended.
        max_seq_length (int): The maximum sequence length the model can handle. Automatically scales RoPE if this exceeds the model's native limit.
        dtype (torch.dtype, optional): Data type for model weights. If None, automatically selects based on GPU capability (BF16 for Ampere+, else FP16).
        load_in_4bit (bool): Enable 4-bit quantization using BitsAndBytes. Significant VRAM savings.
        load_in_8bit (bool): Enable 8-bit quantization.
        load_in_16bit (bool): Load in full half-precision (FP16/BF16). No quantization applied.
        full_finetuning (bool): If True, enables updating all model parameters instead of using LoRA. Requires high VRAM.
        token (str, optional): Hugging Face Hub token for accessing private or gated models.
        device_map (str): Strategy for mapping model layers to devices. Defaults to "sequential".
        rope_scaling (dict, optional): Configuration for extending context length (e.g., {"type": "linear", "factor": 2.0}).
        fix_tokenizer (bool): If True, applies patches to fix common issues with HF tokenizers (missing special tokens, etc.).
        trust_remote_code (bool): Whether to allow custom code from the model repository.
        use_gradient_checkpointing (str or bool): Uses "unsloth" for optimized activation checkpointing to save memory.
        resize_model_vocab (int, optional): New vocabulary size if adding custom tokens.
        revision (str, optional): Specific model revision (branch/commit) to load.
        use_exact_model_name (bool): If True, avoids Unsloth's internal model name mapping.
        offload_embedding (bool): If True, offloads embedding layers to CPU to save GPU VRAM.
        float32_mixed_precision (bool, optional): Forces the use of FP32 mixed precision if enabled.
        fast_inference (bool): If True, enables vLLM backend for faster generation/inference.
        gpu_memory_utilization (float): Fraction of GPU memory to reserve for the vLLM engine (0.0 to 1.0).
        float8_kv_cache (bool): Enables FP8 quantization for the Key-Value cache to support longer context.
        random_state (int): Seed for PRNGs to ensure reproducibility. Default is 3407.
        max_lora_rank (int): The maximum allowed rank (r) for LoRA adapters.
        disable_log_stats (bool): If True, suppresses Unsloth's initialization status messages.
        qat_scheme (str, optional): Scheme for Quantization-Aware Training.
        load_in_fp8 (bool or str): Loads model in 8-bit floating point precision (E4M3/E5M2). Use 'block' for blocked quantization.
        unsloth_tiled_mlp (bool): Experimental optimization for Llama-3 MLP layers to improve speed.

    Returns:
        tuple: (model, tokenizer) where the model is patched with Unsloth kernels.
"""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit", # The BASE model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)



#### LoRA ADAPTERS FOR CPT ##################

"""
Converts a base model into a PeftModel (LoRA) with Unsloth's optimized kernels.

Args:
    model: The pre-trained model to be patched.
    r (int): Rank of the LoRA adapters. Higher = more capacity, more VRAM.
    target_modules (list): List of module names to apply LoRA to.
    lora_alpha (int): Scaling factor for the LoRA weights.
    lora_dropout (float): Dropout probability for LoRA layers (0.0 recommended).
    bias (str): Bias type for LoRA ('none', 'all', 'lora_only').
    use_gradient_checkpointing (str/bool): Use "unsloth" for 2x memory efficiency.
    use_rslora (bool): Whether to use Rank-Stabilized LoRA scaling.
    modules_to_save (list, optional): List of modules to fully train (e.g., 'lm_head').
    init_lora_weights (bool): Whether to initialize adapter weights.
    ensure_weight_tying (bool): Ensures embedding and head weights stay tied if required.

Returns:
    model: The model ready for SFT (Supervised Fine-Tuning).
"""

model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Higher rank is better for learning new knowledge
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        "embed_tokens", "lm_head", # REQUIRED for Continued Pre-training
    ],
    lora_alpha = 32,
    lora_dropout = 0, # Optimized for 0
    bias = "none",    # Optimized for "none"
    use_gradient_checkpointing = "unsloth", # Saves VRAM
    random_state = 3407,
    use_rslora = True, # Rank Stabilized LoRA is often better for CPT
)

####### LOAD DATASET ###########################
dataset = load_dataset("json", data_files="corpus.jsonl", split="train")

####### TRAINER #####################################
"""
Initializes the Supervised Fine-Tuning (SFT) Trainer.

Args:
    model (Union[str, PreTrainedModel]): The model to train, or a path to the model weights.
    args (SFTConfig, optional): Training hyperparameters (learning rate, batch size, output dir).
    data_collator (DataCollator, optional): Function to batch and pad input data.
    train_dataset (Dataset): The dataset used for training.
    eval_dataset (Dataset, optional): The dataset used for evaluation during training.
    processing_class (Tokenizer/Processor): The tool used to encode text or images.
    compute_loss_func (Callable, optional): Custom loss function for training.
    compute_metrics (Callable, optional): Function to calculate metrics (accuracy, etc.) during eval.
    callbacks (list, optional): List of TrainerCallbacks for custom behavior (logging, early stopping).
    optimizers (tuple): A custom (optimizer, scheduler) pair.
    peft_config (PeftConfig, optional): LoRA/PEFT configuration if the model isn't already patched.
    formatting_func (Callable, optional): Function to map dataset columns to a prompt template.
"""

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset, # Your corpus.jsonl from earlier
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = True, # Packs multiple short texts into one block
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 100, # Start small to test
        learning_rate = 5e-5,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()
model.save_pretrained("cpt_lora_model")
tokenizer.save_pretrained("cpt_lora_model")
model.save_pretrained_merged("final_cpt_model", tokenizer, save_method = "merged_16bit")
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method = "q4_k_m")

print(f"Time taken: {trainer_stats.metrics['train_runtime']} seconds")
print(f"Peak VRAM used: {torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024:.2f} GB")