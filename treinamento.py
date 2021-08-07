import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lematizador = WordNetLemmatizer()

intencoes = json.loads(open('intencoes.json').read())

palavras = list()
classes = list()
documentos = list()
ignorados = ['!', '?', '.', ',']

for intencao in intencoes['intencoes']:
    for padrao in intencao['padroes']:
        lista_palavras = nltk.word_tokenize(padrao)
        palavras.extend(lista_palavras)
        documentos.append( (lista_palavras, intencao['categoria']) )

        if intencao['categoria'] not in classes:
            classes.append(intencao['categoria'])

palavras = [lematizador.lemmatize(palavra) for palavra in palavras if palavra not in ignorados]
palavras = sorted(set(palavras))

classes = sorted(set(classes))

pickle.dump(palavras, open('palavras.pk1', 'wb'))
pickle.dump(palavras, open('classes.pk1', 'wb'))

treinamento = list()
output_vazio = [0]*len(classes)

for documento in documentos:
    saco = list()
    palavras_padrao = documento[0]
    palavras_padrao = [lematizador.lemmatize(palavra.lower()) for palavra in palavras_padrao]

    for palavra in palavras:
            saco.append(1) if palavra in palavras_padrao else saco.append(0)

    output_linha = list(output_vazio)
    output_linha[classes.index(documento[1])] = 1
    treinamento.append([saco, output_linha])

random.shuffle(treinamento)
treinamento = np.array(treinamento)

treino_x = list(treinamento[:, 0])
treino_y = list(treinamento[:, 1])

model = Sequential()
model.add(Dense(128, input_shape = (len(treino_x[0]),), activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(treino_y[0]), activation = 'softmax'))

sgd = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

model.fit(np.array(treino_x), np.array(treino_y), epochs = 200, batch_size = 5, verbose = 1)
model.save('chatbot.model.model')

print('Feito!')