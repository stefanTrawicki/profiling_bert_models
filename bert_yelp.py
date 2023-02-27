import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")
model = TFAutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-yelp-polarity", from_pt=True)

def name():
    return "Bert_Yelp"

def run(sentences):
    for s in sentences:
        inputs = tokenizer(s, padding = True, truncation = True, return_tensors='tf')
        outputs = model(**inputs)