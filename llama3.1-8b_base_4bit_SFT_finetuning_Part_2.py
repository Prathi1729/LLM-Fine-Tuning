from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template

# 1. Load your CPT-merged model
# Point this to the folder where you saved "final_cpt_model"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "final_cpt_model", 
    max_seq_length = 4096,
    load_in_4bit = True, # Use 4-bit for speed, or False for 16-bit on MI300X
)

# 2. Add LoRA adapters for the SFT phase
model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Lower rank is fine for SFT as the knowledge is already in the weights
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# 3. Setup the Chat Template (Critical for Llama 3.1)
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

# 4. Format your JSON data
# We assume your JSON looks like: {"instruction": "...", "output": "..."}
def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    outputs      = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        # We wrap your Q&A into the official Llama-3 chat headers
        convo = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
        # apply_chat_template adds the <|begin_of_text|>, <|start_header_id|>, etc.
        text = tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False)
        texts.append(text)
    return { "text" : texts, }

# Load your local JSON file
dataset = load_dataset("json", data_files="your_data.json", split="train")
dataset = dataset.map(formatting_prompts_func, batched = True)

# 5. The SFT Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 4096,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 200, # Adjust based on dataset size
        learning_rate = 2e-4, # Higher than CPT to learn behavior quickly
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "sft_outputs",
    ),
)

trainer.train()

# 6. Final Save
model.save_pretrained_merged("final_assistant_model", tokenizer, save_method = "merged_16bit")