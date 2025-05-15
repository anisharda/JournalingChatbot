from transformers import pipeline

generator = pipeline("text-generation", model="thrishala/mental_health_chatbot")

res = generator(
    "In this course, we will teach you how to",
    max_length = 50,
    #two possible return 
    num_return_sequences=2,
    )


print(res)