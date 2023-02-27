import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-ag-news")
model = TFAutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-ag-news", from_pt=True)

def name():
    return "Roberta_AG_News"

def run(sentences):
    for s in sentences:
        inputs = tokenizer(s, padding = True, truncation = True, return_tensors='tf')
        outputs = model(**inputs)