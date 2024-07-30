import tensorflow as tf
import numpy as np
import pandas as pd
import json
import string
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import os
import pickle
import glob

# チェックポイント保存用のフォルダ
checkpoint_dir = 'checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# モデルとトークナイザーの保存用のフォルダ
models_dir = 'models'
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# データの読み込み
with open('content.json', encoding="utf-8") as content:
    data1 = json.load(content)

# データの整形
tags = []
inputs = []
responses = {}
for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['input']:
        inputs.append(lines)
        tags.append(intent['tag'])

data = pd.DataFrame({"inputs": inputs, "tags": tags})

data['inputs'] = data['inputs'].apply(lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

# トークン化とパディング
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])
x_train = pad_sequences(train)

# ラベルのエンコード
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

# 入力形状と語彙数
input_shape = x_train.shape[1]
vocabulary = len(tokenizer.word_index) + 1  # +1 for padding token
output_length = le.classes_.shape[0]

# モデルの構築
i = Input(shape=(input_shape,))
x = Embedding(vocabulary, 10)(i)
x = LSTM(10, return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length, activation="softmax")(x)
model = Model(i, x)

# モデルのコンパイル
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])

# 最新のチェックポイントを読み込み
def get_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.weights.h5'))
    if checkpoints:
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        return checkpoints[-1]
    return None

latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
if latest_checkpoint:
    print(f'最新のチェックポイント: {latest_checkpoint}')
    model.load_weights(latest_checkpoint)
else:
    print('チェックポイントが見つかりません。最初からトレーニングを開始します。')

# チェックポイントコールバックの作成
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, 'checkpoint_{epoch:02d}.weights.h5'),
    save_weights_only=True,
    save_freq='epoch'
)

# モデルのトレーニング
history = model.fit(x_train, y_train, epochs=200, callbacks=[checkpoint_callback])

# トレーニング後にモデルと関連ファイルを保存
model.save(os.path.join(models_dir, 'chatbot_model.h5'))
with open(os.path.join(models_dir, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)
with open(os.path.join(models_dir, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(le, f)
