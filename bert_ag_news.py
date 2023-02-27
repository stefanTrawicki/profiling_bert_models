import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-ag-news")
model = TFAutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-ag-news", from_pt=True)

# print(tokenizer.decode(output))

def name():
    return "Bert_AG_News"

def run(sentences):
    for s in sentences:
        inputs = tokenizer(s, padding = True, truncation = True, return_tensors='tf')
        outputs = model(**inputs)