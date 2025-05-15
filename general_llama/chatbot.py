from transformers import pipeline, BitsAndBytesConfig



# The model you want to load

model_name = "meta-llama/Llama-3.1-8B"



# Set quantization config

quantization_config = BitsAndBytesConfig(load_in_8bits=True)



# Initialize the pipeline with the quantization config

pipe = pipeline("text-generation", model=model_name, model_kwargs={'quantization_config': quantization_config})



# Your prompt

messages = "Who are you?"



# Generate a response

response = pipe(messages)



# Print the generated response

print(response[0]['generated_text'])

