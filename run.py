# coding: utf-8
import numpy as np

import tensorflow as tf
from keras import backend as K
from keras import layers, models, optimizers, callbacks
from models import CapsNet

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def margin_loss(y_true, y_pred):
	m_plus = 0.9
	m_minus = 0.1
	reg_lambda = 0.5

	L = y_true * tf.square(tf.maximum(0., m_plus - y_pred)) + reg_lambda * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))
	return tf.reduce_mean(tf.reduce_sum(L, axis=1))


def train(model, train, test, save_dir, epochs, batch_size):
	(X_train, y_train) = train
	(X_test, y_test) = test

	# Callbacks.
	logs = callbacks.CSVLogger(filename=save_dir + '/logs.csv')
	checkpoints = callbacks.ModelCheckpoint(filepath=save_dir + '/weights-improvement-{epoch:02d}.hdf5', save_best_only=True, save_weights_only=True, verbose=1)
	early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3,
		restore_best_weights=True, mode='auto')

	model.compile(optimizer=optimizers.Adam(0.001, beta_1=0.7, beta_2=0.999, amsgrad=True),
		loss=[margin_loss], metrics=['accuracy'])

	records = model.fit(X_train, y_train, validation_data=[X_test, y_test], batch_size=batch_size, epochs=epochs, callbacks=[logs, checkpoints, early_stopping], shuffle=True, verbose=1)

	train_loss, train_acc = model.evaluate(X_train, y_train)
	test_loss, test_acc = model.evaluate(X_test, y_test)
	print('----- Training Results -----')
	print('ACC  : ', train_acc)
	print('Loss : ', train_loss)
	print('----- Test Results -----')
	print('ACC  : ', test_acc)
	print('Loss : ', test_loss)

	y_pred = model.predict(X_test)
	y_pred = np.argmax(y_pred, axis=1)
	y_true = np.argmax(y_test, axis=1)

	print('\n--- P / R / F1 ---')
	print(classification_report(y_true, y_pred))
	print('--- Confusion Matrix ---')
	print(confusion_matrix(y_true, y_pred, labels=[0, 1]))

	import csv
	f = open(save_dir+'/outputs.csv', 'w', encoding='utf-8', newline='')
	writer = csv.writer(f)
	writer.writerow(['PRED', 'TRUE'])
	for p, t in zip(y_pred, y_true):
		writer.writerow([str(p), str(t)])
	f.close()

	# print(records.history.keys())

	# Summarize history for acc and loss.
	plt.plot(records.history['accuracy'])
	plt.plot(records.history['val_accuracy'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['training accuracy', 'testing accuracy'], loc='upper left')
	plt.savefig(save_dir + '/model_accuracy.png')
	plt.close()

	# Summarize history for loss
	plt.plot(records.history['loss'])
	plt.plot(records.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['training loss', 'testing loss'], loc='upper left')
	plt.savefig(save_dir + '/model_loss.png')
	plt.close()

	model.save_weights(save_dir + '/trained_model.h5')

if __name__ == '__main__':
	# ### English dataset.
	# from en_data_helpers import load_data
	# import os
	# import numpy as np

	# (X_train, y_train), (X_test, y_test), vocab_size, max_length = load_data('MR')

	## Korean dataset.
	from data_handler import load_nsmc
	from keras.preprocessing import text, sequence
	import itertools

	X_train, y_train, train_max_len = load_nsmc('./nsmc/ratings_train.txt')
	X_test, y_test, test_max_len = load_nsmc('./nsmc/ratings_test.txt')
	print('Train : ', X_train.shape)
	print('Test : ', X_test.shape)

	X = np.concatenate([X_train, X_test])
	Y = np.concatenate([y_train, y_test])
	maxlen = np.max([train_max_len, test_max_len])
	print('Max Len : ', maxlen)

	NUM_WORDS = len(set(list(itertools.chain(*X))))
	print('Num Words : ', NUM_WORDS)
	tokenizer = text.Tokenizer(num_words=NUM_WORDS)
	tokenizer.fit_on_texts(X)

	X_train = tokenizer.texts_to_sequences(X_train)
	X_test = tokenizer.texts_to_sequences(X_test)
	VOCAB_SIZE = len(tokenizer.word_index)

	X_train = sequence.pad_sequences(X_train, maxlen=maxlen, padding='post')
	X_test = sequence.pad_sequences(X_test, maxlen=maxlen, padding='post')

	# Preprocessing for labels (One-hot Encoding).
	from sklearn.preprocessing import LabelEncoder
	from keras.utils import np_utils
	encoder = LabelEncoder()
	encoder.fit(Y)
	encode_train = encoder.transform(y_train)
	encode_test = encoder.transform(y_test)

	y_train = np_utils.to_categorical(encode_train)
	y_test = np_utils.to_categorical(encode_test)

	print('X Train : ', X_train.shape)
	print('Y Train : ', y_train.shape)
	print('X Test : ', X_test.shape)
	print('Y Test : ', y_test.shape)
	print(X_train[0])
	print(y_train[0])

	embedding_dim = 200
	model = CapsNet(input_shape=X_train.shape[1:], n_classes=y_train.shape[-1],
		num_routings=3, vocab_size=NUM_WORDS, embedding_dim=embedding_dim, l2_ratio=0.0001)

	o = 'adam'
	e = 10
	bz = 100

	folder = './Mask_zero_True_o=' + o + '_e=' + str(e) + '_bz=' + str(bz) + '_embedding=' + str(embedding_dim)
	if not os.path.exists(folder):
		os.makedirs(folder)

	# model, train, test, save_dir, epochs, batch_size
	train(model=model, train=(X_train, y_train), test=(X_test, y_test),
		save_dir=folder, epochs=e, batch_size=bz)
