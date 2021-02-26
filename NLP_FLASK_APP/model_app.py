import tensorflow as tf
import numpy as np
import requests
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask,request,jsonify,render_template
app=Flask('category Model')

global category_labels
global labels
category_labels={"beauty":0,"grocery":1,"office":2,"pet":3,"toy":4}
labels=list(category_labels.keys())
new_model=tf.keras.models.load_model('./models/bilstm_model.h5')
with open('./models/tokenizer.pickle','rb') as handle:
    loaded_tokenizer=pickle.load(handle)


def predict_category(txt):
    txt={"tea real delight big fan white tea one best varieties tried flavor subtle overpowering get nice peach flavor sweet yet bitter still taste light flavor smooth white tea celestial seasonings imperial white peach white tea one favorite teas see going back time time really pleasant definitely recommend especially love white tea"}
    seq=loaded_tokenizer.texts_to_sequences(txt)
    padded=pad_sequences(seq,maxlen=80,padding="post",truncating="post")
    pred=new_model.predict(padded) # output [ 5 values  for each class and label number]
    category_labels={"beauty":0,"grocery":1,"office":2,"pet":3,"toy":4}
    labels=list(category_labels.keys())
    print(pred,labels[np.argmax(pred)])
    return labels[np.argmax(pred)]

@app.route('/')
def home():
       return render_template('form.html')

@app.route('/result',methods=['POST'])
def result():
        if request.method== 'POST':
            text=request.form['input']
            predicted_category=predict_category(text)
            return render_template('result.html',text=text,predicted_category=predicted_category)










if __name__=='__main__':
    app.run(host='localhost',port=5001,debug=True)
