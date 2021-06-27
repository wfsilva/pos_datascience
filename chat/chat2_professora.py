# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:44:32 2021
@author: Jasmine Moreira
"""
import pandas as pd
import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Carregar perguntas e resspostas
qdf = pd.read_csv(r'C:\Dados\chat\perguntas.csv', sep=";")
adf = pd.read_csv(r'C:\Dados\chat\respostas.csv', sep=";")

# Converter perguntas para minúsculas
qdf.question = [question.lower() for question in qdf.question]

max_len = 15
max_words = 1000

# Tokenizar e codificar as perguntas
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(qdf.question)
sequences = tokenizer.texts_to_sequences(qdf.question)

# Criar inputs e labels
x_train = pad_sequences(sequences, maxlen=max_len)
y_train = to_categorical(qdf.answer_id)

# Criar o modelo
model = Sequential()
model.add(Embedding(1000, 8, input_length=max_len))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dense(adf.answer_id.max() + 1, activation='softmax'))

opt = optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=1000,
                    batch_size=len(x_train)
                    )

prediction = model(x_train)
print(prediction)
np.argmax(prediction, axis=1)

while (True):
    sentence = input("Você: ")
    sentence = tokenizer.texts_to_sequences([sentence.lower()])
    sentence = pad_sequences(sentence, maxlen=max_len)
    prediction = model(sentence)
    category = np.argmax(prediction, axis=1)[0]
    answer = adf.query('answer_id==' + str(category)).to_numpy()
    print(answer[0][1])