import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("boychaboy/SNLI_roberta-base")
model = TFAutoModelForSequenceClassification.from_pretrained("boychaboy/SNLI_roberta-base", from_pt=True)

def name():
    return "Roberta_SNLI"

def run(sentences):
    for s in sentences:
        inputs = tokenizer(s, padding = True, truncation = True, return_tensors='tf')
        outputs = model(**inputs)