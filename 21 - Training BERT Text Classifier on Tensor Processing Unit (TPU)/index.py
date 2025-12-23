import tensorflow as tf
import logging
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv1D,
    Dropout,
    Input,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from transformers import BertTokenizer, TFBertModel
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
tqdm.pandas()
import re
import random
import matplotlib.pyplot as plt

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)
    
    
max_length = 140
batch_size = 16
dev_size = 0.1
num_class = 4


model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)

train_df = pd.read_csv('../input/aranizi-dailect-training-data/Arabizi-Dailect-Train.csv')

train_df.text = train_df.text.astype(str)
train_df.drop_duplicates(subset=['text'],inplace=True)
train_df.label=train_df.label.map({-1:0,1:2,0:1})
train, dev = train_test_split(train_df, test_size=dev_size, random_state=42)

def bert_encode(data):
    tokens = tokenizer.batch_encode_plus(
        data, max_length=max_length, padding="max_length", truncation=True
    )
    return tf.constant(tokens["input_ids"])
train_encoded = bert_encode(train.text)
dev_encoded = bert_encode(dev.text)
train_labels = tf.keras.utils.to_categorical(train.label.values, num_classes=num_class)
dev_labels = tf.keras.utils.to_categorical(dev.label.values, num_classes=num_class)
train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_encoded, train_labels))
    .shuffle(100)
    .batch(batch_size)
).cache()
dev_dataset = (
    tf.data.Dataset.from_tensor_slices((dev_encoded, dev_labels))
    .shuffle(100)
    .batch(batch_size)
).cache()

def bert_tweets_model():
    bert_encoder = TFBertModel.from_pretrained(model_name, output_attentions=True)
    input_word_ids = Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    last_hidden_states = bert_encoder(input_word_ids)[0]
    clf_output = Flatten()(last_hidden_states)
    net = Dense(512, activation="relu")(clf_output)
    net = Dropout(0.3)(net)
    net = Dense(440, activation="relu")(net)
    net = Dropout(0.3)(net)
    output = Dense(num_class, activation="softplus")(net)
    model = Model(inputs=input_word_ids, outputs=output)
    return model

with strategy.scope():
    model = bert_tweets_model()
    adam_optimizer = Adam(learning_rate=1e-5)
    model.compile(
        loss="categorical_crossentropy", optimizer=adam_optimizer, metrics=["accuracy"]
    )
    model.summary()
    

history = model.fit(
    train_dataset,
    batch_size=batch_size,
    epochs=5,
    validation_data=dev_dataset,
    verbose=1,
)

model.save_weights('weights.h5', overwrite=True)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.show()
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

test = pd.read_csv("../input/arabizi-dialect/Test (1).csv")
test.text = test.text.astype(str)
test_encoded = bert_encode(test.text)
##Loading Test Data
test_dataset = tf.data.Dataset.from_tensor_slices(test_encoded).batch(batch_size)
## Prediction on test Datasets
predicted_tweets = model.predict(test_dataset, batch_size=batch_size)
predicted_tweets_binary = np.argmax(predicted_tweets, axis=-1)
## Submisssion 
my_submission = pd.DataFrame({"ID": test.ID, "label": predicted_tweets_binary})
my_submission.label = my_submission.label.map({1: -1, 3: 1, 2: 0})
my_submission.to_csv("/kaggle/working/my_submission.csv", index=False)

my_submission.label.value_counts()
