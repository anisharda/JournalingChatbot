from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline("sentiment-analysis")

res = classifier("I do not not love my life!")

print(res)

# model_name="distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# classifier = pipeline("sentiment-analysis", model =model, tokenizer=tokenizer)

# res = classifier("I love my life!")

# print(res)
