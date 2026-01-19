from unsloth import FastLanguageModel, PatchDPOTrainer
from trl import ORPOConfig, ORPOTrainer
from datasets import load_dataset
import torch

# 1. Load your SFT-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "final_assistant_model", 
    max_seq_length = 4096,
    load_in_4bit = True, 
)

# 2. Add Alignment Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
)

# 3. Format the Preference Data
def format_orpo(examples):
    # ORPO expects 'prompt', 'chosen', and 'rejected' columns
    # We wrap them in the Llama-3.1 chat template
    prompt = [f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{p}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" for p in examples["prompt"]]
    chosen = [f"{c}<|eot_id|>" for c in examples["chosen"]]
    rejected = [f"{r}<|eot_id|>" for r in examples["rejected"]]
    return { "prompt" : prompt, "chosen" : chosen, "rejected" : rejected }


"""[
  {
    "prompt": "What is the primary cooling requirement mentioned in the technical manual?",
    "chosen": "The manual states the system requires a liquid cooling flow rate of 5.5L/min at 20°C.",
    "rejected": "The system just needs standard air cooling from fans."
  }
]"""
dataset = load_dataset("json", data_files="preference_data.json", split="train")
dataset = dataset.map(format_orpo, batched = True)

# 4. Initialize the ORPO Trainer
# PatchORPOTrainer is an Unsloth optimization for TRL's trainer
trainer = ORPOTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    args = ORPOConfig(
        learning_rate = 8e-6,        # Very low for alignment to prevent "forgetting"
        lr_scheduler_type = "linear",
        max_steps = 100,             # Alignment is usually fast
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        beta = 0.1,                  # The "Alignment Strength" (0.1 is standard)
        max_prompt_length = 512,
        max_length = 1024,
        bf16 = True,                 # Native for MI300X
        optim = "adamw_8bit",
        output_dir = "final_aligned_outputs",
    ),
)

trainer.train()

# 5. FINAL EXPORT (The finished product)
model.save_pretrained_merged("final_production_model", tokenizer, save_method = "merged_16bit")