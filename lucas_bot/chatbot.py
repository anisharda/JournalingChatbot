# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the model and tokenizer
model_name = "lucas-w/mental-health-chatbot-1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize a text-generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Example prompt for the chatbot
prompt = "I'm feeling really anxious today, can you help?"

# Generate a response
output = generator(prompt, max_length=100, num_return_sequences=1)

# Print the output
print(output[0]["generated_text"])
