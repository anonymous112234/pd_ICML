import keras_tuner
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

import os
import csv

# cd Documents/
# cd populationDescent/
# python3 -m venv ~/venv-metal
# source ~/venv-metal/bin/activate
# python3 -m 

# grad_steps = 25 trials * 2 executions each trial * 782 batches per execution + (5 * 782) for final training = 43000 steps


# PTB dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Download the Penn Treebank dataset
path = tf.keras.utils.get_file('ptb-tiny', 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt')

# Read the data
with open(path, 'r') as f:
    text = f.read()

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1

# Create input sequences using the tokenizer
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Pad sequences
max_sequence_length = max([len(x) for x in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Create input and output data
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Split the data into training and testing sets
train_images, X_temp, train_labels, y_temp = X[:int(len(X)*.8)], X[int(len(X)*.8):], y[:int(len(y)*.8)], y[int(len(y)*.8):]
validation_images, test_images, validation_labels, test_labels = X_temp[:int(len(X_temp)*.5)], X_temp[int(len(X_temp)*.5):], y_temp[:int(len(y_temp)*.5)], y_temp[int(len(y_temp)*.5):]

# print(len(train_images), len(test_images), len(validation_labels)) = 676460, 84558, 84558


def build_model(hp):
	total_words = 9650
	max_sequence_length = 98
	
	model = keras.Sequential()
	model.add(tf.keras.layers.Embedding(input_dim=total_words, output_dim=50, input_length=max_sequence_length-1))

	# RandomSearch for regularization rate
	hp_reg = hp.Float("reg_term", min_value=1e-5, max_value=1e-1)

	model.add(tf.keras.layers.LSTM(655, kernel_regularizer=l2(l=hp_reg)))
	model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

	# RandomSearch for learning rate
	hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
	
	# compile model
	model.compile(optimizer=keras.optimizers.legacy.Adam(learning_rate=hp_learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

	return model
 


# seed:
def set_seeds(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	tf.random.set_seed(seed)
	np.random.seed(seed)

def set_global_determinism(seed):
	set_seeds(seed=seed)

	os.environ['TF_DETERMINISTIC_OPS'] = '1'
	os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

	tf.config.threading.set_inter_op_parallelism_threads(1)
	tf.config.threading.set_intra_op_parallelism_threads(1)



SEED = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]

for seed in SEED:  


	set_global_determinism(seed=seed)
	print(seed), print("") 

	import time
	start_time = time.time()  


	max_trials = 25
	model_num = "1 with reg"


	# define tuner
	print("random search")
	tuner = keras_tuner.RandomSearch(
	    hypermodel=build_model,
	    objective="val_accuracy",
	    max_trials=max_trials,
	    executions_per_trial=2,
	    overwrite=True,
	    project_name="CIFAR: %s" % SEED
	)


	# search
	tuner.search(train_images, train_labels, validation_data=(validation_images, validation_labels), batch_size=64)

	# retrieve and train best model
	best_hps = tuner.get_best_hyperparameters(5)
	model = build_model(best_hps[0])


	# Use early stopping
	callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)


	# TRAIN Model
	print("")
	print("TRAINING")
	train_epochs = 20
	model.fit(train_images, train_labels, batch_size=64, validation_data=(validation_images, validation_labels), epochs=train_epochs, callbacks=[callback])


	time_lapsed = time.time() - start_time


	# evaluating model on test and train data
	batch_size = 64

	np.random.seed(0)
	eIndices = np.random.choice(84558 - 1, size = (batch_size*25, ), replace=False)
	random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels = train_images[eIndices], train_labels[eIndices], test_images[eIndices], test_labels[eIndices]

	print(""), print(""), print("Evaluating models on test data after randomization")

	# evaluating on train, test images
	lossfn = tf.keras.losses.SparseCategoricalCrossentropy()
	# train_loss = lossfn(random_batch_train_labels, model(random_batch_train_images))
	# test_loss = lossfn(random_batch_test_labels, model(random_batch_test_images))

	train_loss = model.evaluate(random_batch_train_images, random_batch_train_labels)[0]
	test_loss = model.evaluate(random_batch_test_images, random_batch_test_labels)[0]

	print("unnormalized train loss: %s" % train_loss)
	print("unnormalized test loss: %s" % test_loss)
	# print("normalized (1/1+loss) test loss: %s" % ntest_loss)



	model_num = "1_with_reg"


	# writing data to excel file
	data = [[test_loss, train_loss, model_num, max_trials, time_lapsed, seed]]

	with open('../KT_RandomSearch_PTB_Benchmark_with_regularization.csv', 'a', newline = '') as file:
	    writer = csv.writer(file)
	    writer.writerows(data)

