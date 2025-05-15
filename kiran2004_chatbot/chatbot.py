from transformers import pipeline

from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel

#load tokenizer and model
model_name = "Kiran2004/GPT2_MentalHealth_ChatBot"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

#generate text
generator = pipeline("text-generation", model=model, tokenizer=tokenizer,device=0)
prompt = "How is you day?"
output = generator(prompt, max_length=150, num_return_sequences=1)
print(output[0]["generated_text"])
