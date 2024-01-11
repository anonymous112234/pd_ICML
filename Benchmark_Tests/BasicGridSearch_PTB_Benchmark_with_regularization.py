# use Python 3.9
# python3.9 -m venv env
# source new3.9/bin/activate

import random
import math
import matplotlib.pyplot as plt
import scipy
from scipy.special import softmax
import numpy as np

# Typing
import typing
from typing import TypeVar, Generic
from collections.abc import Callable

from tqdm import tqdm
import statistics
import dataclasses
from dataclasses import dataclass
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models
#import keras.backend as K

import tensorflow as tf

import csv


from PTB_models import new_hps_NN_individual_with_regularization


# cd Documents
# cd pd_ICLR_rebuttal
# python3 -m venv ~/venv-metal
# source ~/venv-metal/bin/activate
# python3 -m BasicGridSearch_PTB_Benchmark_with_regularization


# setting seed
SEED = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]


import os
def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


for h in range(len(SEED)):
	s = SEED[h]
	print(""), print("SEED:"), print(SEED[h]), print("")
	set_global_determinism(seed=SEED[h])

	population, reg_list, model_num = new_hps_NN_individual_with_regularization()


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


	# observing optimization progress
	# unnormalized
	def observer(NN_object, tIndices):
		random_batch_test_images, random_batch_test_labels = test_images[tIndices], test_labels[tIndices]

		lossfn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
		test_loss = lossfn(random_batch_test_labels, NN_object(random_batch_test_images))

		ntest_loss = 1/(1+test_loss)

		return test_loss

	def graph_history(history, trial, model_string, training_loss_data_string, test_loss_data_string, best_lr_data, best_reg_amount_string):
		integers = [i for i in range(1, (len(history))+1)]
		x = [j * rr for j in integers]
		y = history

		plt.scatter(x, history, s=20)

		plt.title("PD CIFAR10")
		plt.tight_layout()
		# plt.savefig("TEST_DATA/HP_trial_%s.png" % trial)
		plt.show(block=True), plt.pause(0.5), plt.close()


	# Observation
	observer_history = []
	rr = 1

	trial = 1


	# PARAMETERS
	iterations = 30

	batch_size = 64
	batches = 128
	epochs = 1

	gradient_steps = iterations * epochs * batches * (len(population))


	graph = False


	import time
	start_time = time.time()

	with tf.device('/device:GPU:0'):
		# TRAINING
		for i in tqdm(range(iterations)):

			indices = np.random.choice(676460 - 1, size = (batch_size*batches, ), replace=False)
			vIndices = np.random.choice(84558 - 1, size = (batch_size*10, ), replace=False)

			random_batch_train_images, random_batch_train_labels = train_images[indices], train_labels[indices]
			random_batch_validation_images, random_batch_validation_labels = validation_images[vIndices], validation_labels[vIndices]

			# indices
			oIndices = np.random.choice(84558 - 1, size = (batch_size*1, ), replace=False)
			population_training_losses = []

			for j in range(len(population)):

				print("model %s" % (j+1))
				population[j].fit(random_batch_train_images, random_batch_train_labels, validation_data = (random_batch_validation_images, random_batch_validation_labels), epochs=epochs, verbose=1, batch_size=batch_size)

				print("regularization_amount: %s" % reg_list[j])
				print("learning rate: %s" % population[j].optimizer.learning_rate)
				print("")

				# population_training_losses.append(training_loss)

				# observing optimization progress
				if (i%rr)==0:
						if i!=(iterations-1):
							individual_observer_loss = observer(population[j], oIndices)
							population_training_losses.append(individual_observer_loss)


			if (i%rr)==0:
				if population_training_losses:
					population_training_losses = np.array(population_training_losses)
					observer_history.append(np.min(population_training_losses))
					population_training_losses = []

			# if (i%rr)==0:
			# 		if i!=(iterations-1):
			# 			print(""), print("observing"), print("..."), print("")
			# 			population_training_losses = np.array(population_training_losses)
			# 			observer_history.append(np.min(population_training_losses))



	time_lapsed = time.time() - start_time


	with tf.device('/device:GPU:0'):
		# # Evaluating on test data
		np.random.seed(0)
		eIndices = np.random.choice(4999, size = (batch_size*25, ), replace=False)
		random_batch_train_images, random_batch_train_labels, random_batch_test_images, random_batch_test_labels = train_images[eIndices], train_labels[eIndices], test_images[eIndices], test_labels[eIndices]

		training_losses, evaluation_losses, evaluation_accuracies = [], [], []

		for h in range(len(population)):
			print("model %s" % (h+1))

			training_loss, training_accuracy = population[h].evaluate(random_batch_train_images, random_batch_train_labels, batch_size = batch_size)
			test_loss, test_acc = population[h].evaluate(random_batch_test_images, random_batch_test_labels, batch_size = batch_size)

			# ntest_loss = 1/(1+test_loss)
			# ntest_loss = np.array(ntest_loss)

			training_losses.append(training_loss)

			evaluation_losses.append(test_loss)
			evaluation_accuracies.append(test_acc)


		best_training_model_loss_unnormalized = np.min(training_losses)

		best_test_model_loss = np.min(evaluation_losses)
		best_index = evaluation_losses.index(best_test_model_loss)

		best_lr = (population[best_index]).optimizer.learning_rate
		best_reg_amount = reg_list[best_index]

		evaluation_losses = np.array(evaluation_losses)

		test_loss_data = statistics.mean(evaluation_losses)
		test_acc_data = statistics.mean(evaluation_accuracies)



	# printing all data to console
	print("")
	model_string = "model #%s" % (best_index+1)
	print(model_string)
	training_loss_data_string = "unnormalized train loss of best model %s" % best_training_model_loss_unnormalized
	print(training_loss_data_string)
	test_loss_data_string = "unnormalized test loss of best model: %s" % best_test_model_loss
	# best_test_loss_unnormalized = ((1/best_test_model_loss)-1)
	print(test_loss_data_string)
	best_lr_data = "best LR: %s" % best_lr
	print(best_lr_data)
	best_reg_amount_string = "best reg amount: %s" % best_reg_amount
	print(best_reg_amount_string), print("")


	# writing data to excel file
	data = [[best_test_model_loss, best_training_model_loss_unnormalized, gradient_steps, model_num, best_reg_amount, best_lr, iterations, epochs, batches, batch_size, time_lapsed, s]]

	with open('../BasicGridSearch_PTB_Benchmark_with_regularization.csv', 'a', newline = '') as file:
		writer = csv.writer(file)
		writer.writerows(data)

	# graphing data
	if graph:
		graph_history(observer_history, trial, model_string, training_loss_data_string, test_loss_data_string, best_lr_data, best_reg_amount_string)



