import tensorflow as tf

# mod = tf.keras.models.load_model('tf_model.h5')

# print(mod.evaluate(["Stocks rallies and the british pound gained"], ["positive"], verbose=2))

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")
model = TFAutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb", from_pt=True)

# def pred_to_sentiment(outputs):
#     sentiments = [
#         "negative",
#         "positive"
#     ]

#     i = max(range(len(outputs[0][0])), key=outputs[0][0].__getitem__)
#     return sentiments[i]
    
def name():
    return "Bert_IMDB"

def run(sentences):
    for s in sentences:
        inputs = tokenizer(s, padding = True, truncation = True, return_tensors='tf')
        outputs = model(**inputs)