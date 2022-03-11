# -*- coding: utf-8 -*-
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras import backend as K
from capsule_model import CapsuleNetworks

from sklearn.metrics import classification_report, confusion_matrix

import time
from termcolor import colored
import matplotlib.pyplot as plt

# for GPU.
os.environ["CUDA_VISIBLE_DEVICES"] ="0, 1"

def padding_text(inputs, max_len):
	outputs = list()
	for text in inputs:
		_after = list(text[:max_len])

		while len(_after) < max_len:
			_after.append(0) # padding character is 0.
		outputs.append(_after)
	return np.array(outputs)

def one_hot(train, test):
	labels = sorted(list(set(np.concatenate((train, test)))))
	one_hot = np.zeros((len(labels), len(labels)), int)
	np.fill_diagonal(one_hot, 1)
	label_map = dict(zip(labels, one_hot))

	train = np.eye(len(label_map))[train]
	test = np.eye(len(label_map))[test]
	return train, test

def margin_loss(y_true, y_pred):
	m_plus = 0.9
	m_minus = 0.1
	_lambda = 0.5

	L = y_true * K.square(K.maximum(0., m_plus - y_pred)) + _lambda * (1 - y_true) * K.square(K.maximum(0., y_pred - m_minus))
	return K.mean(K.sum(L, 1))

def step_decay(epoch):
	init_lr = 0.01
	drop = 0.5
	epochs_drop = 5
	lr = init_lr * tf.math.pow(drop, tf.math.floor((1+epoch)/epochs_drop))
	return lr

def training(model, train, test, epochs, batch_size, save_path):
	(X_train, y_train) = train
	(X_test, y_test) = test

	# Callbacks.
	ckpt_path = os.path.join(save_path, '{}_checkpoint.h5'.format('IMDB'))
	ckpt_callback = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True,
		save_weights_only=True, mode='max')
	early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, mode='auto')
	_s = step_decay
	lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=_s, verbose=1)

	# Training-Process.
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.7, beta_2=0.999, amsgrad=True),
		loss=[margin_loss], metrics=['accuracy'])

	record = model.fit(X_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs,
		callbacks=[ckpt_callback, early_stop, lr_decay], shuffle=True, verbose=1)

	model.load_weights(ckpt_path)
	scores = model.evaluate(X_test, y_test)
	print(colored(scores, 'green'))
	print('Test: Acc {:.4f} / Loss {:.6f}'.format(scores[-1], scores[0]))

if __name__ =='__main__':
	### Load the data: IMDB (Movie review for sentiment classification).
	num_words = 10000 # Only consider the top 10k words.
	max_len = 200 # Only consider the first 200 words of each movie review.

	(X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=num_words)

	## Padding to IMDB data.
	X_train = padding_text(inputs=X_train, max_len=max_len)
	X_test = padding_text(inputs=X_test, max_len=max_len)
	y_train, y_test = one_hot(train=y_train, test=y_test)
	print('Number of Train: ', X_train.shape[0])
	print('Number of Valid: ', X_test.shape[0])
	print('Number of Class: ', y_train.shape[-1])
	print('Number of Class: ', y_test.shape[-1])

	# Check the save path.
	save_path = './CapsuleNet_trained/'
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	# parameters.
	e = 50
	bz = 256
	embedding_dim = 300

	model = CapsuleNetworks(X=X_train, y=y_train, routing=True, num_routings=3, vocab_size=num_words, embedding_dim=embedding_dim)
	model.summary()
	time.sleep(5)

	# Training.
	training(model=model, train=(X_train, y_train), test=(X_test, y_test), epochs=e, batch_size=bz, save_path=save_path)
