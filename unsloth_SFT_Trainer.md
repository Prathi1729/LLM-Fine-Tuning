# Core Data & Model Parameters
model: The model you previously initialized (e.g., via FastLanguageModel). It can also be a string path to a model, though in Unsloth, you usually pass the patched model object.

args: This takes TrainingArguments or SFTConfig. This is where you define learning rate, batch size, number of epochs, and logging steps.

train_dataset / eval_dataset: The data the model learns from and the data used to check performance. Usually a Hugging Face Dataset object.

processing_class: (Formerly tokenizer). This handles the conversion of text into numbers. In modern versions, "processing_class" is used because it supports both tokenizers (text) and processors (vision/audio).

# Customizing the Training Logic
data_collator: A function that forms "batches" of data. It handles padding (making all sequences the same length) and creating "labels" for the model to predict.

compute_loss_func: Allows you to override how the model calculates its error. If None, it defaults to standard Cross-Entropy loss.

formatting_func: Crucial for SFT. This function takes a raw row from your dataset and converts it into a specific prompt template (like Alpaca or ShareGPT).

compute_metrics: A function to calculate accuracy, F1 score, or BLEU during evaluation. It converts the model's raw output into human-readable performance stats.

# Optimization & Infrastructure
optimizers: A tuple containing a custom optimizer (like AdamW) and a learning rate scheduler. If None, the trainer creates them based on your args.

optimizer_cls_and_kwargs: Allows you to pass a specific optimizer class (like bitsandbytes.optim.AdamW8bit) and its settings directly.

callbacks: A list of objects that can "hook" into the training process to do things like trigger early stopping or log data to Weights & Biases.

peft_config: If you didn't already convert your model to LoRA using get_peft_model, you can pass the configuration here, and the trainer will do it for you.

preprocess_logits_for_metrics: A performance-saving function. It processes the model's huge output tensor before sending it to the evaluation function to avoid Out-of-Memory (OOM) errors.