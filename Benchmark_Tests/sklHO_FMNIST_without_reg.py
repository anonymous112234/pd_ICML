import keras_tuner
import tensorflow as tf
from tensorflow import keras

from keras.callbacks import TensorBoard

import numpy as np
import random

import os
import csv


from hyperopt import fmin, tpe, hp


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
# python3 -m keras_tuner_FMNIST_without_reg_test

# grad_steps = 25 trials * 2 executions each trial * 782 batches per execution + (5 * 782) for final training = 43000 steps



# Fashion-MNIST dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

sample_shape = train_images[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1)

# Reshape data
train_images = train_images.reshape(len(train_images), input_shape[0], input_shape[1], input_shape[2])
test_images  = test_images.reshape(len(test_images), input_shape[0], input_shape[1], input_shape[2])

# normalizing data
train_images, test_images = train_images / 255.0, test_images / 255.0

# splitting data into validation/test set
validation_images, validation_labels = test_images[0:5000], test_labels[0:5000]
test_images, test_labels = test_images[5000:], test_labels[5000:]



def build_model(params):
    
	model = models.Sequential()

    model.add(layers.Conv2D(filters=params['conv1_filters'], kernel_size=params['conv1_kernel'], strides=(2, 2),
                            dilation_rate=(1, 1), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.Conv2D(filters=params['conv2_filters'], kernel_size=params['conv2_kernel'], strides=(2, 2),
                            dilation_rate=(1, 1), activation='relu'))
    model.add(layers.Conv2D(filters=params['conv3_filters'], kernel_size=params['conv3_kernel'],
                            dilation_rate=(1, 1), activation='relu'))

    model.add(layers.Flatten(input_shape=(28, 28)))

    model.add(layers.Dense(units=params['dense_units'], activation="relu"))
    model.add(layers.Dropout(params['dropout']))
    model.add(layers.Dense(10, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=params['learning_rate'])


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
# SEED = [101, 150, 200]



for seed in SEED:  

	set_global_determinism(seed=seed)
	print(seed), print("") 

	import time
	start_time = time.time()  


	max_trials = 25
	model_num = "FMNIST 4"


	# Define the search space
	space = {
	    'conv1_filters': hp.choice('conv1_filters', [32, 64, 128]),
	    'conv1_kernel': hp.choice('conv1_kernel', [3, 5]),
	    'conv2_filters': hp.choice('conv2_filters', [64, 128, 256]),
	    'conv2_kernel': hp.choice('conv2_kernel', [3, 5]),
	    'conv3_filters': hp.choice('conv3_filters', [128, 256, 512]),
	    'conv3_kernel': hp.choice('conv3_kernel', [3, 5]),
	    'dense_units': hp.choice('dense_units', [128, 256, 512]),
	    'dropout': hp.uniform('dropout', 0.2, 0.5),
	    'learning_rate': hp.loguniform('learning_rate', -4, -2),
	}

	def objective(params):
	    model = create_cnn(params)

	    # Train and evaluate the model (modify this according to your dataset and training process)
	    train_epochs = 5
	    history = model.fit(train_images, train_labels, batch_size= 64, validation_data=(validation_images, validation_labels), epochs=train_epochs, callbacks=[callback])
	    
		# Access the validation accuracy
		validation_accuracy = history.history['val_accuracy'][-1]

	    # Hyperopt minimizes the objective function, so negate the metric you want to maximize
	    return -validation_accuracy


	# Run Hyperopt to find the best hyperparameters
	best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=10)

	# retrieve and train best model
	model = build_model(best)

	# Use early stopping
	callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=0)

	# TRAIN Model
	print("")
	print("TRAINING")
	train_epochs = 20
	hist = model.fit(train_images, train_labels, batch_size= 64, validation_data=(validation_images, validation_labels), epochs=train_epochs, callbacks=[callback])

	# getting history
	# print("history"), print(hist.history["val_loss"])
	grad_steps = [i * 936 for i in hist.history['val_loss']]
	# print(""), print("grad_steps"), print(grad_steps)


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



	model_num = "4_without_reg"


	# writing data to excel file
	data = [[test_loss, train_loss, model_num, max_trials, time_lapsed, seed]]

	with open('../sklHO_FMNIST_without_reg.csv', 'a', newline = '') as file:
	    writer = csv.writer(file)
	    writer.writerows(data)





