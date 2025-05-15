import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import PeftModel

import os



os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



# 1. BitsAndBytes 8-bit config

bnb_config = BitsAndBytesConfig(

    load_in_8bit=True,

    llm_int8_enable_fp32_cpu_offload=True

)



# 2. Load base model

base_model = AutoModelForCausalLM.from_pretrained(

    "meta-llama/Llama-2-13b-chat-hf",

    device_map="auto",

    quantization_config=bnb_config,

    trust_remote_code=True

)



# 3. Load fine-tuned adapter

model = PeftModel.from_pretrained(

    base_model,

    "/work/pi_jaimedavila_umass_edu/asharda_umass_edu/LLM_scripts/llama_2b/Trained Model2"

)

model.eval()



# 4. Load tokenizer

tokenizer = AutoTokenizer.from_pretrained(

    "meta-llama/Llama-2-13b-chat-hf",

    use_fast=False

)



# 5. Construct prompt clearly (avoid short/incomplete prompts)

user_input = (

    "[INST] I feel really alone lately. Even when I’m with friends or family, it’s like no one truly sees me. [/INST]"

)



inputs = tokenizer(

    user_input,

    return_tensors="pt",

    truncation=True,

    max_length=2048,

).to(model.device)



# 6. Generate

num_responses = 3

with torch.no_grad():

    outputs = model.generate(

        **inputs,

        max_new_tokens=1024,

        do_sample=True,

        temperature=0.8,

        top_p=0.9,

        num_return_sequences=num_responses,

        eos_token_id=tokenizer.eos_token_id,

        pad_token_id=tokenizer.eos_token_id

    )



# 7. Decode and print

print("\n=== Model Responses ===")

for i, output in enumerate(outputs):

    decoded = tokenizer.decode(output, skip_special_tokens=True)

    stripped = decoded.replace(user_input, "").strip()

    print(f"\nResponse {i+1}:\n{stripped}")

