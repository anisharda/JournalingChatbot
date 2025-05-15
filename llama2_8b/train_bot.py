from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig, DataCollatorForSeq2Seq)

from datasets import load_dataset

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training



import torch



print("STARTING the code")



# 1. Load Dataset

full_dataset = load_dataset("empathetic_dialogues", trust_remote_code=True, cache_dir="/work/pi_jaimedavila_umass_edu/asharda_umass_edu/hf_cache")



print("step one completed, dataset loaded")



# 2. Load Model + Tokenizer with 4-bit quantization

model_name = "vibhorag101/llama-2-13b-chat-hf-phr_mental_therapy"



bnb_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_use_double_quant=True,

    bnb_4bit_compute_dtype=torch.float16,

    bnb_4bit_quant_type="nf4",

)





tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token



print("tokenize the model")



model = AutoModelForCausalLM.from_pretrained(

    model_name,

    quantization_config=bnb_config,

    device_map="auto"

)





print("model loaded + tokenization = step 2 done")





# 3. Prepare model for LoRA fine-tuning

print("prep model for fine tuning")



model = prepare_model_for_kbit_training(model)



print("model prepped")



# LoRA config (can be tuned later)

print("doing lora config")

lora_config = LoraConfig(

    r=8,

    lora_alpha=16,

    target_modules=["q_proj", "v_proj"],  # Common for LLaMA models

    lora_dropout=0.1,

    bias="none",

    task_type="CAUSAL_LM",

)



print("get peft model about to run")

model = get_peft_model(model, lora_config)



print("step 3 done")



# 4. Preprocess Function

print("start preprocessing step")



def preprocess_function(examples):



    inputs = [p + " " + u for p, u in zip(examples['prompt'], examples['utterance'])]

    targets = [u for u in examples['utterance']]



    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    # to account for mismatch issue, switch max length to 512

    # likely will slow model due to extraneous padding to match sizes

    model_inputs['labels'] = labels['input_ids']

    return model_inputs



print("function created for step 4")



# 5. Tokenize Dataset

dataset = {



    "train": full_dataset["train"],



    "validation": full_dataset["validation"]  # Optional — include if you want to evaluate



}







# Tokenize both sets



tokenized_datasets = {



    split: ds.map(preprocess_function, batched=True)



    for split, ds in dataset.items()



}







print("tokenized dataset done")





print("tokenized dataset done")



train_dataset = tokenized_datasets["train"]

print("dataset tokenized")

eval_dataset = tokenized_datasets["validation"]

print("eval dataset tokenized")





# 6. Data Collator

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")



print("step 6 done")



# 7. Training Arguments



print("prep for step 7")



training_args = TrainingArguments(



    output_dir='./results_final',

    num_train_epochs=10,

    per_device_train_batch_size=8,                # A100 can easily handle 16+

    gradient_accumulation_steps=2,                 # No need to accumulate with large batches

    per_device_eval_batch_size=16,                 # Speed up evaluation too

    warmup_steps=100,

    weight_decay=0.01,

    logging_dir='./logs',

    logging_steps=10,

    save_strategy="epoch",                         # Save once per epoch

    evaluation_strategy="epoch", # Changed to "epoch" to match save_strategy

    eval_steps=1000,                               # Evaluate every 1000 steps.  This is irrelevant now.

    fp16=True,

    save_total_limit=2,

    report_to="none",

    ddp_find_unused_parameters=False,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    gradient_checkpointing=True,       # Enable gradient checkpointing

)



print("finish step 7")



# 8. Trainer Setup



print("trainer set up")

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    tokenizer=tokenizer,

    data_collator=data_collator,

)

print("trainer thing set")



# 9. Train with LoRA

print("start the training")

trainer.train()

print("training is completed")



print("successfully trained!")



# 10. Save only the LoRA adapter weights!

model.save_pretrained("/work/pi_jaimedavila_umass_edu/asharda_umass_edu/LLM_scripts/llama2_7b/Trained Models")

tokenizer.save_pretrained("/work/pi_jaimedavila_umass_edu/asharda_umass_edu/LLM_scripts/llama2_7b/Trained Models")



print("stuff is saved")
