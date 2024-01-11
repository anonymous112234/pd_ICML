import numpy as np

from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models
# from tensorflow.keras.layers import CuDNNLSTM

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split


from collections import namedtuple
# namedtuple individual "class"
NN_Individual = namedtuple("NN_Individual", ["nn", "opt_obj", "LR_constant", "reg_constant"])

# Testing population descent
def new_pd_NN_individual_without_regularization():
	total_words = 9650
	max_sequence_length = 98

	model_num = "1"
	model = tf.keras.Sequential([
	tf.keras.layers.Embedding(input_dim=total_words, output_dim=50, input_length=max_sequence_length-1),
	tf.keras.layers.BatchNormalization(),
	tf.keras.layers.LSTM(100),
	tf.keras.layers.Dense(total_words, activation='softmax')
    ])

	optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3) # 1e-3 (for FMNIST)
	LR_constant = 10**(np.random.normal(-4, 2))
	reg_constant = 10**(np.random.normal(0, 2))

	# creating NN object with initialized parameters
	NN_object = NN_Individual(model, optimizer, LR_constant, reg_constant)

	return NN_object, model_num

def new_pd_NN_individual_with_regularization():
	total_words = 9650
	max_sequence_length = 98

	model_num = "1"
	model = tf.keras.Sequential([
	tf.keras.layers.Embedding(input_dim=total_words, output_dim=50, input_length=max_sequence_length-1),
	tf.keras.layers.LSTM(100, kernel_regularizer=l2(0.01)),
	tf.keras.layers.Dense(total_words, activation='softmax')
    ])

	optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001) # 1e-3 (for FMNIST)
	LR_constant = 10**(np.random.normal(-4, 2))
	reg_constant = 10**(np.random.normal(0, 2))

	# creating NN object with initialized parameters
	NN_object = NN_Individual(model, optimizer, LR_constant, reg_constant)

	# Print the model summary
	# model.summary()


	return NN_object, model_num


# Testing Hyperparameter search
def new_hps_NN_individual_without_regularization():
	
	regularization_amount = [0]
	learning_rate = [0.001]


	population = []
	reg_list = []

	for r in range(len(regularization_amount)):
		for l in range(len(learning_rate)):
			total_words = 9650
			max_sequence_length = 98

			model = tf.keras.Sequential([
			tf.keras.layers.Embedding(input_dim=total_words, output_dim=50, input_length=max_sequence_length-1),
			tf.keras.layers.LSTM(100, kernel_regularizer=l2(0.1)),
			tf.keras.layers.Dense(total_words, activation='softmax')
		    ])


			optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate[l])

			# Compile the model
			model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

			population.append(model)
			reg_list.append(regularization_amount[r])

	population = np.array(population)

	model_num = "1"
	return population, reg_list, model_num


# Testing Hyperparameter search
def new_hps_NN_individual_with_regularization():
	total_words = 9650
	max_sequence_length = 98

	learning_rate = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
	regularization_amount = [0.01, 0.001, 0.0001, 0.00001, 0.000001]


	population = []
	reg_list = []

	for r in range(len(regularization_amount)):
		for l in range(len(learning_rate)):
			total_words = 9650
			max_sequence_length = 98

			model = tf.keras.Sequential([
			tf.keras.layers.Embedding(input_dim=total_words, output_dim=50, input_length=max_sequence_length-1),
			tf.keras.layers.LSTM(100, kernel_regularizer=l2(0.01)),
			tf.keras.layers.Dense(total_words, activation='softmax')
		    ])



			optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate[l])

			model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

			population.append(model)
			reg_list.append(regularization_amount[r])

	population = np.array(population)


	return population, reg_list, model_num




