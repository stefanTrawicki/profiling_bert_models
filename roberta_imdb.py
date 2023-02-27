import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-imdb")
model = TFAutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-imdb", from_pt=True)

def name():
    return "Roberta_IMDB"

def run(sentences):
    for s in sentences:
        inputs = tokenizer(s, padding = True, truncation = True, return_tensors='tf')
        outputs = model(**inputs)