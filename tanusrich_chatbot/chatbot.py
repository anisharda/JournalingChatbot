from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = "tanusrich/Mental_Health_Chatbot"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move the model to the appropriate device (CPU or GPU)
device = torch.device("cpu")
model.to(device)

# Generate a response
def generate_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example interaction
user_input = "I'm feeling lonely and anxious. What can I do?"
response = generate_response(user_input)
print("Chatbot: ", response)
