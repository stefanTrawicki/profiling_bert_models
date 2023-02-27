import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("VictorSanh/roberta-base-finetuned-yelp-polarity")
model = TFAutoModelForSequenceClassification.from_pretrained("VictorSanh/roberta-base-finetuned-yelp-polarity", from_pt=True)

def name():
    return "Roberta_Yelp"

def run(sentences):
    for s in sentences:
        inputs = tokenizer(s, padding = True, truncation = True, return_tensors='tf')
        outputs = model(**inputs)