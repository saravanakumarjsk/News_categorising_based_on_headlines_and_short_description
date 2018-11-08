import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from keras.layers import Dense, Input, Bidirectional, Activation, Conv1D, GRU
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Flatten, SpatialDropout1D
from keras.layers import GlobalAveragePooling1D, concatenate
from keras.layers import Reshape, merge, Concatenate, Lambda, Average
from keras.models import Sequential, Model
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import np_utils

from sklearn.model_selection import train_test_split


# load data

df = pd.read_json(
    'C:\\Users\\jana\\Desktop\\new project\\DeepResearch-master\\Hierarchical_Attention_Network\\News_Category_Dataset\\News_Category_Dataset.json', lines=True)
df.head()
cates = df.groupby('category')
print("total categories:", cates.ngroups)
print(cates.size())

# In the above category there are two WORLDPOST so they should be merged into one, thus a lambda function is used

df.category = df.category.map(
    lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)

# using headlines and short_description as input X

df['text'] = df.headline + " " + df.short_description

# tokenizing

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)
X = tokenizer.texts_to_sequences(df.text)
df['words'] = X

# delete some empty and short data

df['word_length'] = df.words.apply(lambda i: len(i))
df = df[df.word_length >= 5]

df.head()

# using 50 for padding length

maxlen = 50
X = list(sequence.pad_sequences(df.words, maxlen=maxlen))
print(X[0])

# converting the category to id

# grouping the categories to a list
categories = df.groupby('category').size().index.tolist()
category_int = {}  # assign a mapping var for cat to int
int_category = {}  # assign a mapping var for int to cat
for i, k in enumerate(categories):
    category_int.update({k: i})  # assigning values from enum as cat ->  int
    int_category.update({i: k})  # vice versa

df['char2id'] = df['category'].apply(lambda x: category_int[x])


# just for some visual understanding of the data
categories = df.groupby('category').size().index.tolist()
print(categories[9], '\n', df['headline'][9])
print(df['char2id'][:10])

# glove embedding is one of the most usefull model for representing distributed word representation.
word_index = tokenizer.word_index
emb_dim = 100
embeddings_index = {}
# use a pretrained model
f = open('C:\\Users\\jana\\Desktop\\new project\\DeepResearch-master\\Hierarchical_Attention_Network\\glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s unique tokens.' % len(word_index))

# create an embedding_matrix and prepare the embedding layer for LSTM operation
embedding_matrix = np.zeros((len(word_index) + 1, emb_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
# i/p dim = len(word_index)
# o/p dim = emb_dim = 100
embedding_layer = Embedding(len(word_index) + 1, emb_dim, embeddings_initializer=Constant(
    embedding_matrix), input_length=maxlen, trainable=False)
print(embedding_matrix)
len(word_index)
emb_dim

# prepared dataset
X = np.array(X)  # convert value to array for further calculation.
Y = np_utils.to_categorical(list(df.char2id))  # one-hot array encoding format.

# and split to training set and validation set
x_train, x_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=0)
print(x_train)
print('\n')
print(x_val)


#  Bidirectional GRU(LSTM) + Conv nets for better results

# Using convolution nets as the input layer to the text data for better recognization of sentiment from the text
# and this will be passed to the  LSTM(GRU) layer for further analysis of data.

# CONV NETS FOR FEATURE RECOG.

# max length is 50 which is the padding value.
inp = Input(shape=(maxlen,), dtype='int32')
# this embeddign layer turns positive integers into dense vectors for calculation
x = embedding_layer(inp)
x = SpatialDropout1D(0.2)(x)

# set the return_seq to True for returning the values into the lstm cell again.

x = Bidirectional(GRU(128, return_sequences=True, dropout=0.1,
                      recurrent_dropout=0.1))(x)  # LSTM LAYER
x = Conv1D(64, kernel_size=3)(x)  # CONV LAYER
avg_pool = GlobalAveragePooling1D()(x)  # AVERAGE POOLING FOR TIME SERIES DATA
max_pool = GlobalMaxPooling1D()(x)
x = concatenate([avg_pool, max_pool])
outp = Dense(len(int_category), activation="softmax")(x)

BiGRU = Model(inp, outp)
BiGRU.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['acc'])

BiGRU.summary()

# training

bigru_history = BiGRU.fit(x_train,
                          y_train,
                          batch_size=128,
                          epochs=20,
                          validation_data=(x_val, y_val))

acc = bigru_history.history['acc']
val_acc = bigru_history.history['val_acc']
loss = bigru_history.history['loss']
val_loss = bigru_history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.title('Accuracy')
plt.plot(epochs, acc, 'green', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Loss')
plt.plot(epochs, loss, 'green', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()

plt.show()

# to calculate the accuracy of the model.


def evaluate_accuracy(model):
    predicted = model.predict(x_val)
    diff = y_val.argmax(axis=-1) - predicted.argmax(axis=-1)
    corrects = np.where(diff == 0)[0].shape[0]
    total = y_val.shape[0]
    return float(corrects / total)


print("model Bidirectional GRU + Conv:  %.3f*100" % evaluate_accuracy(BiGRU))
