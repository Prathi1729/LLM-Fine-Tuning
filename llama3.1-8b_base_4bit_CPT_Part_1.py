from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import torchvision

max_seq_length = 4096 # Supports RoPE Scaling automatically
dtype = None # None for auto detection. Float16 for Tesla T4, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization

# Set AMD Architecture override
import os
# Disable Unsloth's version guard
os.environ["UNSLOTH_VERSION_CHECK"] = "0"

# Ensure the MI300X architecture is correctly recognized
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "9.4.2"
os.environ["ROCM_PATH"] = "/opt/rocm"

print(f"Torch version: {torch.__version__}")
print(f"TorchVision version: {torchvision.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")

from unsloth import FastLanguageModel
import torch

# Check if it works now
print(f"Torch version: {torch.__version__}")
print(f"Is ROCm available: {torch.cuda.is_available()}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit", # The BASE model
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)



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
# model.save_pretrained_merged("model_merged", tokenizer, save_method = "merged_16bit")
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method = "q4_k_m")

print(f"Time taken: {trainer_stats.metrics['train_runtime']} seconds")
print(f"Peak VRAM used: {torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024:.2f} GB")


#### TO Test the model convert to GGUF
"""
File Will be in llama.cpp

python3 convert_hf_to_gguf.py ../model_merged --outfile ../model.f16.gguf
"""


### ADDITIONAL 

"""
Quantize to 4-bit (Q4_K_M)

# Move to the build folder where your binaries are
cd build/bin/

# Run the quantization
./llama-quantize ../../model.f16.gguf ../../model.q4_k_m.gguf Q4_K_M

"""