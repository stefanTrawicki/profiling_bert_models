import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = TFAutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# def pred_to_sentiment(outputs):
#     sentiments = [
#         "positive",
#         "neutral",
#         "negative"
#     ]

#     i = max(range(len(outputs[0][0])), key=outputs[0][0].__getitem__)
#     return sentiments[i]
    

# print(outputs[0][0])
# print(pred_to_sentiment(outputs))

def name():
    return "Finbert"

def run(sentences):
    for s in sentences:
        inputs = tokenizer(s, padding = True, truncation = True, return_tensors='tf')
        outputs = model(**inputs)