from transformers import pipeline

# Load the model from Hugging Face (set device to -1 for CPU)
model_name = "thrishala/mental_health_chatbot"
chatbot = pipeline("text-generation", model=model_name, device=0)  # device=-1 ensures CPU is used

#function to interact with the chatbot
def chat_with_bot(prompt):
    print("User: " + prompt)
    print("has entered chat with bot function")
    response = chatbot(prompt, max_length=150, num_return_sequences=1)
    print("Bot: " + response[0]["generated_text"])

# Example of interaction
if __name__ == "__main__":
    while True:
        print("is inside function")
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Bot: Goodbye!")
            break
        chat_with_bot(user_input)
