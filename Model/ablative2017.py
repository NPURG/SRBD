import csv
from keras import models
import tensorflow as tf
from keras.layers import Bidirectional, GaussianDropout, LSTM, MultiHeadAttention
from tensorflow.python import layers
from tensorflow.keras import Input, Model
from tensorflow.python.keras.layers import Embedding, GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate


from keras import backend as K

from keras import initializers, regularizers, constraints
from tensorflow.python.layers.base import Layer
import numpy as np
from keras.utils.np_utils import to_categorical

import tensorflow.compat.v1 as tf1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random

embedding_size = 300  
n_feature = 27  
f_vector = 5  
epochs = 2000  
batch_size = 64  
validation_split = 0.1


def f1_score(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)  
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)  
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)  
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)  

    
    p = tp / (tp + fp + K.epsilon())  
    r = tp / (tp + fn + K.epsilon())  

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf1.is_nan(f1), tf.zeros_like(f1),
                  f1)  
    return f1


def precision(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf1.is_nan(f1), tf.zeros_like(f1), f1)
    return p


def recall(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf1.is_nan(f1), tf.zeros_like(f1), f1)
    return r


class Attention_layer(Layer):
    """
        Attention operation, with a context/query vector, for temporal data.
        Supports Masking.
        Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
        "Hierarchical Attention Networks for Document Classification"
        by using a context vector to assist the attention
        
            3D tensor with shape: `(samples, steps, features)`.
        
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(AttentionWithContext())
        """

    def __init__(self,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention_layer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        super(Attention_layer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)

        a = K.exp(uit)

        
        if mask is not None:
            
            a *= K.cast(mask, K.floatx())

        
        
        
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        print(a)
        
        print(x)
        weighted_input = x * a
        print(weighted_input)
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


class Self_Attention(tf.keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(3, input_shape[2], self.output_dim), initializer='uniform',
                                      trainable=True)
        super(Self_Attention, self).build(input_shape)

    def call(self, x):
        WQ = K.dot(x, self.kernel[0])
        WK = K.dot(x, self.kernel[1])
        WV = K.dot(x, self.kernel[2])
        
        

        QK = K.batch_dot(WQ, K.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64 ** 0.5)
        QK = K.softmax(QK)
        
        V = K.batch_dot(QK, WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "output_dim": self.output_dim
        })
        return config


basePath2015 = "D:/Documents/dataSet/bots/Cresci et al. Database (2015)/DataAnalysis/"
basePath2017 = "D:/Documents/dataSet/bots/Cresci et al. Database (2017)/datasets_full/DataAnalysis/"
basePathHoneypot = "D:/Documents/dataSet/bots/social_honeypot_icwsm_2011/honeypot/DataAnalysis/"
readBasePath = "ReadabilityAnalysis/ReadabilityValue/"
senmanticBasePath = "SenmanticsAnalysis/SenmanticsValue/"
sentenceBasePath = "JaroWinklerSimilarity/JaroWinklerSimilarityValue/"
sentimentBasePath = "SentimentAnalysis/SentimentAnalysisValue/"
timeGapBasePath = "TimeAnalysis/TimeGapValue/"
urlBasePath = "UrlCountAnalysis/"


files2015 = ["E13/", "TFP/", "FSF/", "INT/", "TWT/"]
labels = ["readability_textstat_MAX.csv", "readability_textstat_MIN.csv", "readability_textstat_MEAN.csv",
          "readability_textstat_MEDIAN.csv", "readability_textstat_MODE.csv"]

filenames2015 = ["TFP", "TWT"]

filenames2017 = ["genuine_accounts", "social_spambots_2"]

filenamesHoney = ["legitimate", "pollute"]



def randint_generation(min, max, mount):
    list = []
    while len(list) != mount:
        unit = random.randint(min, max)
        if unit not in list:
            list.append(unit)
    return list



def download_data():
    
    Featrues_Positive = {}
    
    Featrues_Negative = {}
    Featrues = [Featrues_Positive, Featrues_Negative]
    for k in range(2):
        print('k = ', k)
        
        for i in range(len(labels)):
            print("open the file ", filenames2017[k])
            max_feature = {}
            min_feature = {}
            mean_feature = {}
            mode_feature = {}
            median_feature = {}
            featrues = [max_feature, min_feature, mean_feature, median_feature, mode_feature]
            for i in range(5):
                with open(basePath2017 + readBasePath + filenames2017[k] + '/' + labels[i], "r", encoding="utf-8") as f:
                    print(labels[i])
                    reader = csv.DictReader(f)
                    for row in reader:
                        if row['id'] not in featrues[i]:
                            featrues[i][row['id']] = [float(row["Syllable Count"]),
                                                      float(row["Monosyllable Count"]),
                                                      float(row["Polysyllable Count"]), float(row["Lexicon Count"]),
                                                      float(row["Sentence Count"]),
                                                      float(row["Character Count"]),
                                                      float(row["Letter Count"]),
                                                      float(row["The Flesch Reading Ease formula"]),
                                                      float(row["The Flesch-Kincaid Grade Level"]),
                                                      float(row["The Fog Scale"]),
                                                      float(row["The SMOG Index"]),
                                                      float(row["The Coleman-Liau Index"]),
                                                      float(row["Automated Readability Index"]),
                                                      float(row["Linsear Write Formula"]),
                                                      float(row["Dale-Chall Readability Score"]),
                                                      float(row["Readability Consensus based upon all the above tests"]),
                                                      float(row["Spache Readability Formula"]),
                                                      float(row["McAlpine EFLAW Readability Score"]),
                                                      float(row["Reading Time"]),
                                                      float(row["Difficult Words"])
                                                      ]
                        else:
                            featrues[i][row['id']].append(
                                [float(row["Syllable Count"]),
                                 float(row["Monosyllable Count"]),
                                 float(row["Polysyllable Count"]), float(row["Lexicon Count"]),
                                 float(row["Sentence Count"]),
                                 float(row["Character Count"]), float(row["Letter Count"]),
                                 float(row["The Flesch Reading Ease formula"]),
                                 float(row["The Flesch-Kincaid Grade Level"]), float(row["The Fog Scale"]),
                                 float(row["The SMOG Index"]),
                                 float(row["The Coleman-Liau Index"]),
                                 float(row["Automated Readability Index"]),
                                 float(row["Linsear Write Formula"]),
                                 float(row["Dale-Chall Readability Score"]),
                                 float(row["Readability Consensus based upon all the above tests"]),
                                 float(row["Spache Readability Formula"]),
                                 float(row["McAlpine EFLAW Readability Score"]),
                                 float(row["Reading Time"]),
                                 float(row["Difficult Words"])
                                 ])
        row_labels = ["Syllable Count", "Monosyllable Count", "Polysyllable Count", "Lexicon Count",
                      "Sentence Count",
                      "Character Count", "Letter Count", "The Flesch Reading Ease formula",
                      "The Flesch-Kincaid Grade Level", "The Fog Scale",
                      "The SMOG Index", "The Coleman-Liau Index", "Automated Readability Index",
                      "Linsear Write Formula",
                      "Dale-Chall Readability Score",
                      "Readability Consensus based upon all the above tests", "Spache Readability Formula",
                      "McAlpine EFLAW Readability Score", "Reading Time", "Difficult Words"]
        for i in range(len(row_labels)):
            for key in max_feature:
                Featrues[k][key] = [
                    [float(max_feature[key][0]), float(min_feature[key][0]), float(mean_feature[key][0]),
                     float(median_feature[key][0]), float(mode_feature[key][0])]]
                for i in range(1, len(row_labels)):
                    Featrues[k][key].append(
                        [float(max_feature[key][i]), float(min_feature[key][i]), float(mean_feature[key][i]),
                         float(median_feature[key][i]), float(mode_feature[key][i])])

        print(len(Featrues[0]))

        
        with open(basePath2017 + senmanticBasePath + filenames2017[k] + '.csv', "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['id'] in Featrues[k]:
                    Featrues[k][row['id']].append(
                        [float(row['result_max']), float(row['result_min']), float(row['result_mean']),
                         float(row['result_mode']), float(row['result_mode'])])

        
        with open(basePath2017 + sentimentBasePath + filenames2017[k] + '.csv', "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['id'] in Featrues[k]:
                    Featrues[k][row['id']].append(
                        [float(row['max_neg']), float(row['min_neg']), float(row['mean_neg']), float(row['median_neg']),
                         float(row['zhongshu_neg'])])
                    Featrues[k][row['id']].append(
                        [float(row['max_neu']), float(row['min_neu']), float(row['mean_neu']), float(row['median_neu']),
                         float(row['zhongshu_neu'])])
                    Featrues[k][row['id']].append(
                        [float(row['max_pos']), float(row['min_pos']), float(row['mean_pos']), float(row['median_pos']),
                         float(row['zhongshu_pos'])])
                    Featrues[k][row['id']].append(
                        [float(row['max_compound']), float(row['min_compound']), float(row['mean_compound']),
                         float(row['median_compound']), float(row['zhongshu_compound'])])

        
        with open(basePath2017 + sentenceBasePath + filenames2017[k] + '.csv', "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['id'] in Featrues[k]:
                    Featrues[k][row['id']].append([float(row['TweetSimilarityMax']), float(row['TweetSimilarityMIN']),
                                                   float(row['TweetSimilarityMEAN']),
                                                   float(row['TweetSimilarityMEDIAN']),
                                                   float(row['TweetSimilarityMODE'])])

        
        
        
        
        
        
        

        
        with open(basePath2017 + urlBasePath + filenames2017[k] + '.csv', "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['user_id'] in Featrues[k]:
                    Featrues[k][row['user_id']].append(
                        [float(row['proportion']), float(row['proportion']), float(row['proportion']),
                         float(row['proportion']), float(row['proportion'])])

        print('k = ', k)
        print(type(Featrues[k]))
        print("311----- len(Featrues) = ", len(Featrues[k]))
        

    
    ID_positives = []
    for key in Featrues_Positive:
        ID_positives.append(key)
    
    ID_negatives = []
    for key in Featrues_Negative:
        ID_negatives.append(key)

    print("ID_positives = ", len(ID_positives))
    print("ID_negatives = ", len(ID_negatives))

    ll = min(len(ID_positives), len(ID_negatives))
    list_positive = randint_generation(0, len(ID_positives) - 1, ll)
    list_negative = randint_generation(0, len(ID_negatives) - 1, ll)

    print("list_positive = ", len(list_positive))
    print("list_negative = ", len(list_negative))

    feature_Embeddings = []
    data_labels = []
    for i in range(ll):
        if (len(Featrues_Positive[ID_positives[list_positive[i]]]) == n_feature and len(
                Featrues_Negative[ID_negatives[list_negative[i]]]) == n_feature):
            feature_Embeddings.append(Featrues_Positive[ID_positives[list_positive[i]]])
            data_labels.append([int(1)])
            feature_Embeddings.append(Featrues_Negative[ID_negatives[list_negative[i]]])
            data_labels.append([int(0)])

    print("len(feature_Embeddings) = ", len(feature_Embeddings))
    print("len(feature_Embeddings[0]) = ", len(feature_Embeddings[0]))
    print("featrue_Embedding[0] = ", feature_Embeddings[0])
    print("len(data_labels) = ", len(data_labels))
    print("len(data_labels[0]) = ", len(data_labels[0]))

    feature_Embeddings = np.array(feature_Embeddings)
    print(feature_Embeddings.shape)
    data_labels = np.array(data_labels)
    print(data_labels.shape)
    print("type(feature_Embeddings) = ", type(feature_Embeddings))
    print("type(data_labels) = ", type(data_labels))

    return feature_Embeddings, data_labels




def get_data(n, input_dim, attention_column=1):
    x = np.random.standard_normal(size=(n, input_dim, 32))
    y = np.random.randint(low=0, high=2, size=(n, 1))

    print("type(x) = ", type(x))
    print("type(y) = ", type(y))
    print("x.shape = ", x.shape)
    print("y.shape = ", y.shape)
    return x, y


def build_modle():
    S_inputs = Input(shape=(n_feature, f_vector))
    print("S_input.shape = ", S_inputs.shape)

    
    
    
    attention = MultiHeadAttention(num_heads=8, name='Multi-Head', key_dim=4)(S_inputs, S_inputs)
    
    
    
    
    
    
    
    print("attention.shape = ", attention.shape)
    conc = tf.reshape(attention, [-1, 135])
    
    x = tf.keras.layers.Dropout(0.5)(conc)
    x = tf.keras.layers.Dense(embedding_size, activation='tanh')(conc)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(embedding_size, activation='tanh')(
        x)  
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(embedding_size, activation='tanh')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(embedding_size, activation='tanh')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)

    model = Model(inputs=S_inputs, outputs=outputs)
    model.summary()
    
    
    Adam = tf.keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0,
                                    amsgrad=False)
    
    loss = 'categorical_crossentropy'
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, verbose=2)
    
    model.compile(loss=loss, optimizer=Adam, metrics=["accuracy", precision, recall, f1_score], run_eagerly=True)
    
    x_train, y_train = download_data()
    y_train = to_categorical(y_train, 2)
    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    print(type(x_train))
    
    
    model.fit(np.array(x_train), np.array(y_train), validation_split=validation_split, epochs=epochs,
              batch_size=batch_size, verbose=2, callbacks=[early_stopping])

    score = model.evaluate(x_train, np.array(y_train), batch_size=batch_size)
    print("score = ", score)
    preds = model.predict(x_train, batch_size=batch_size, verbose=1)
    print("preds =", preds)

    print('epochs = ', epochs, ', batch_size = ', batch_size, ', validation_split = ', validation_split)
    
    model.save('./my_model.h5')


if __name__ == "__main__":
    
    
    
    
    build_modle()
    
    
    
    
    
    
    
    
    