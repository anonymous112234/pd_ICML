import keras_tuner
import tensorflow as tf
from tensorflow import keras
import numpy as np
import random

import os
import csv


# GC
from google.colab import auth
from google.cloud import storage

# Authenticate with Google Cloud
auth.authenticate_user()

project_id = 'schedulesktrsfmnist'
bucket_name = 'schedulesktrsfmnist'

client = storage.Client(project=project_id)
bucket = client.get_bucket(bucket_name)



# cd Documents/
# cd populationDescent/
# python3 -m venv ~/venv-metal
# source ~/venv-metal/bin/activate
# python3 -m keras_tuner_CIFAR_new

# grad_steps = 25 trials * 2 executions each trial * 782 batches per execution + (5 * 782) for final training = 43000 steps



# CIFAR10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# normalizing data
train_images, test_images = train_images / 255.0, test_images / 255.0

validation_images, validation_labels = test_images[0:5000], test_labels[0:5000]
test_images, test_labels = test_images[5000:], test_labels[5000:]




def build_model(hp):

	hp_reg = hp.Float("reg_term", min_value=1e-5, max_value=1e-1)

	model = keras.Sequential()
	model.add(tf.keras.layers.Conv2D(32,  kernel_size = 3, activation='relu', input_shape = (32, 32, 3), kernel_regularizer=tf.keras.regularizers.l2(l=hp_reg)))
	model.add(tf.keras.layers.Conv2D(64,  kernel_size = 3, activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(128,  kernel_size = 3, activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((2, 2)))
	model.add(tf.keras.layers.Conv2D(64,  kernel_size = 3, activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D((4, 4)))

	model.add(tf.keras.layers.Flatten())


	model.add(tf.keras.layers.Dense(256, activation = "relu", kernel_regularizer=tf.keras.regularizers.l2(l=hp_reg)))
	model.add(tf.keras.layers.Dense(10, activation = "softmax"))


	# Tune the learning rate schedule
	lr_schedule = hp.Choice('learning_rate_schedule', values=['exponential', 'inverse_time', 'polynomial'])
    

	if lr_schedule == 'exponential':
		lr = hp.Float('initial_learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
		decay_rate = hp.Float('decay_rate', min_value=0.8, max_value=0.99, default=0.9)
		decay_steps = hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000, default=5000)
		lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate)
		optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
	elif lr_schedule == 'inverse_time':
		lr = hp.Float('initial_learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
		decay_rate = hp.Float('decay_rate', min_value=0.1, max_value=0.9, default=0.5)
		decay_steps = hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000, default=5000)
		lr_schedule = keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=lr, decay_steps=decay_steps, decay_rate=decay_rate)
		optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
	elif lr_schedule == 'polynomial':
		lr = hp.Float('initial_learning_rate', min_value=1e-4, max_value=1e-2, sampling='log', default=1e-3)
		end_learning_rate = hp.Float('end_learning_rate', min_value=1e-6, max_value=1e-2, sampling='log', default=1e-6)
		decay_steps = hp.Int('decay_steps', min_value=1000, max_value=10000, step=1000, default=5000)
		power = hp.Float('power', min_value=0.1, max_value=2.0, default=1.0)
		lr_schedule = keras.optimizers.schedules.PolynomialDecay(
		    initial_learning_rate=lr, decay_steps=decay_steps, end_learning_rate=end_learning_rate, power=power
		)
		optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)


    # Compile the model
	model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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

	# tf.config.threading.set_inter_op_parallelism_threads(1)
	# tf.config.threading.set_intra_op_parallelism_threads(1)



SEED = [5, 15, 24, 34, 49, 60, 74, 89, 97, 100]

for seed in SEED:  


	set_global_determinism(seed=seed)
	print(seed), print("") 

	import time
	start_time = time.time()  


	max_trials = 25
	model_num = "6 without reg"


	# define tuner
	print("random search")
	tuner = keras_tuner.RandomSearch(
	    hypermodel=build_model,
	    objective="val_accuracy",
	    max_trials=max_trials,
	    executions_per_trial=2,
	   	directory='gs://{}/reg_CIFAR10_tuner_results{}'.format(bucket_name, seed),
	    project_name='schedules_KTRS_CIFAR10'
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
	model.fit(train_images, train_labels, batch_size= 64, validation_data=(validation_images, validation_labels), epochs=train_epochs, callbacks=[callback])


	time_lapsed = time.time() - start_time


	# evaluating model on test and train data
	batch_size = 64

	np.random.seed(0)
	eIndices = np.random.choice(4999, size = (batch_size*25, ), replace=False)
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



	model_num = "6_without_reg"


	# writing data to excel file
	data = [[test_loss, train_loss, model_num, max_trials, time_lapsed, seed]]

	with open('../schedules_KTRS_CIFAR_without_reg.csv', 'a', newline = '') as file:
	    writer = csv.writer(file)
	    writer.writerows(data)

