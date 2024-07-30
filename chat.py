import random
import string
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json
import os

# モデルとトークナイザーの読み込み
models_dir = 'models'
model_path = os.path.join(models_dir, 'chatbot_model.h5')
tokenizer_path = os.path.join(models_dir, 'tokenizer.pkl')
label_encoder_path = os.path.join(models_dir, 'label_encoder.pkl')

model = tf.keras.models.load_model(model_path)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
with open(label_encoder_path, 'rb') as f:
    le = pickle.load(f)

# 応答辞書の読み込み
with open('content.json', encoding="utf-8") as content:
    data1 = json.load(content)
responses = {}
for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']

# 会話の開始
while True:
    prediction_input = input('You : ')  # ユーザーの入力を取得
    # 句読点を削除し、小文字に変換
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    
    # テキストをトークン化し、パディング
    prediction_input_seq = tokenizer.texts_to_sequences([prediction_input])
    prediction_input_padded = pad_sequences(prediction_input_seq, maxlen=model.input_shape[1])
    
    # モデルからの出力を取得
    output = model.predict(prediction_input_padded)
    output = output.argmax()  # 最大値のインデックスを取得
    
    # タグを取得
    response_tag = le.inverse_transform([output])[0]
    
    # 応答を選択
    response = random.choice(responses.get(response_tag, ["申し訳ありません、理解できませんでした。"]))
    print("AI : ", response)
    
    # 「goodbye」タグの場合、会話を終了
    if response_tag == "goodbye":
        break
