import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/roberta2roberta_L-24_gigaword")
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/roberta2roberta_L-24_gigaword", from_pt=True)



def run(sentence):
    inputs = tokenizer(sentence, padding = True, truncation = True, return_tensors='tf')
    output = model(**inputs)
    print(output)