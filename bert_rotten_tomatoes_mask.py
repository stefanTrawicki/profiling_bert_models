import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-rotten_tomatoes", mask_token="[MASK]")
model = TFAutoModelForMaskedLM.from_pretrained("textattack/bert-base-uncased-rotten_tomatoes", from_pt=True)

def name():
    return "Bert_Rotten_Tomatoes"

def run(sentence):
    inputs = tokenizer(sentence, padding = True, truncation = True, return_tensors='tf')
    output = model(**inputs)

    logits = output.logits[0, -1, :]
    softmax = tf.math.softmax(logits, axis=-1)
    argmax = tf.math.argmax(softmax, axis=-1)
    # print(sentence, "[", tokenizer.decode(argmax), "]")