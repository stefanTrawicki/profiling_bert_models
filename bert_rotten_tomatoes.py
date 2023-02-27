import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-rotten-tomatoes")
model = TFAutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-rotten-tomatoes", from_pt=True)

def name():
    return "Bert_Rotten_Tomatoes"

def run(sentences):
    for s in sentences:
        inputs = tokenizer(s, padding = True, truncation = True, return_tensors='tf')
        outputs = model(**inputs)