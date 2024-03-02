import os
import torch
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame
from keras.utils.np_utils import to_categorical
from sklearn.metrics.pairwise import cosine_similarity
from keras import backend as K
from sklearn.metrics import f1_score
import tensorflow.python.compat.v1 as tf1

from Model.keras_gat import GraphAttention

tweets_account = 200
N = feature_np.shape[0]  
F = feature_np.shape[1]  
n_classes = 2  
F_ = 32  
embedding_size = 300  
n_attn_heads = 9  
dropout_rate = 0.5  
l2_reg = 5e-4 / 2
learning_rate = 5e-3  
es_patience = 100  
epochs = 300  
batch_size = 16  



def get_data():
    data  = []
    return data


def build_model():
    
    X_in = tf.keras.layers.Input(shape=(F,))
    A_in = tf.keras.layers.Input(shape=(N,))


    x = tf.keras.layers.Dropout(X_in, dropout=dropout_rate)
    
    graph_attention_1 = GraphAttention(F_,
                                       attn_heads=n_attn_heads,
                                       attn_heads_reduction='concat',
                                       dropout_rate=dropout_rate,
                                       activation='relu',
                                       kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
                                       attn_kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
    GAT_Embedding = graph_attention_1([x, A_in])

    dropout2 = tf.keras.layers.Dropout(dropout_rate)
    GAT_Embedding = dropout2(GAT_Embedding)
    print("dropout2 --- GAT_Embedding:")
    print(np.array(GAT_Embedding).shape)

    result2 = GAT_Embedding

    x = tf.keras.layers.Dense(embedding_size, activation='relu')(gru2)  
    x = tf.keras.layers.Dropout(0.5)(x)  
    x = tf.keras.layers.Dense(embedding_size, activation='relu')(
        x)  
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(embedding_size, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    y = tf.keras.layers.Dense(2, activation='softmax')(x)  

    model = tf.keras.models.Model(inputs=input, outputs=y)  
    return model


if __name__=="__main__":
    model = build_model()
    model.summary()
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


