

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Input

from tensorflow.keras import Model


class TextCnnRand(Model):

    def __init__(self, vocabulary_size):
        super(TextCnnRand, self).__init__()
        self.embedding = Embedding(vocabulary_size, 16)
        self.pooling = GlobalAveragePooling1D()
        self.dense = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.pooling(x)
        return self.dense(x)



