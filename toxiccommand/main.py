import tensorflow as tf 

from tensorflow.keras.layers import  TextVectorization

import os 
import matplotlib.pyplot as plt 
import tensorflow as tf
import pandas as pd 
import numpy as np
import tensorflow as tf
import gradio as gr


model = tf.keras.models.load_model('toxicity_model(12).h5')
df = pd.read_csv(os.path.join('ToxicCommandDataset','train','train.csv'))
X = df["comment_text"]
y = df.drop(columns=["comment_text","id"], axis=1)
y = y.values
MAX_WORDS = 200000  #number of words in the vocab
vectorizer = TextVectorization(max_tokens=MAX_WORDS , #number of words in the vocab
    output_sequence_length = 1800 , #it the length of the outsequense  
    output_mode = 'int'#the word number is integear
    )

vectorizer.adapt(X.values)

# input_text = input("Enter Your comment")

# pred =model.predict(np.expand_dims(input_text,0))
# (pred > 0.5).astype(int)
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(df.columns[2:]):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)
    print(comment)
    
    return text

interface = gr.Interface(fn=score_comment, 
                         inputs=gr.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')
interface.launch(share=True)