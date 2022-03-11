# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import backend as K

from capsule_layers import CapsuleBlock, CapsuleNorm

def CapsuleNetworks(X, y, routing, num_routings, vocab_size, embedding_dim, training=True):
	'''
	Implementation of Capsule networks for Text Classificatoin,
	which was proposed in "Text classification using capsules (Kim et al., 2020)".
	[https://www.sciencedirect.com/science/article/pii/S0925231219314092]

	- Reference: https://github.com/TeamLab/text-capsule-network
	'''

	# Input layer.
	inputs = tf.keras.layers.Input(shape=X.shape[1:])
	embeddings = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
		mask_zero=True, trainable=True)(inputs)
	embeddings = tf.keras.layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(embeddings) # Reshape sequence for convolution.

	# Convolution layer with Non-linear gate (ELU).
	elu_layer = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, embedding_dim), strides=1,
		use_bias=False, activation=None)(embeddings)
	elu_layer = tf.keras.layers.BatchNormalization()(elu_layer)
	elu_layer = tf.keras.layers.Activation('elu')(elu_layer)
	# Convolution layer.
	conv_layer = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, embedding_dim), strides=1,
		use_bias=False, activation=None)(embeddings)
	conv_layer = tf.keras.layers.BatchNormalization()(conv_layer)

	# Gate layer.
	gate_layer = tf.keras.layers.Multiply()([elu_layer, conv_layer])
	gate_layer = tf.keras.layers.Dropout(rate=0.1)(gate_layer, training=training)

	# Primary Capsule layer.
	primary_caps = tf.keras.layers.Conv2D(filters=6*10, kernel_size=(K.int_shape(gate_layer)[1], 1),
		strides=1, use_bias=False, activation=None)(gate_layer)
	primary_caps = tf.keras.layers.Reshape((6, 10))(primary_caps)
	primary_caps = tf.keras.layers.BatchNormalization()(primary_caps)
	primary_caps = tf.keras.layers.Activation('relu')(primary_caps)

	# Capsule layer.
	capsule = CapsuleBlock(num_capsule=y.shape[-1], dim_capsule=16, routing=routing, num_routings=num_routings)(primary_caps)

	# Output layer with CapsuleNorm.
	outputs = CapsuleNorm(name='outputs')(capsule)

	model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
	return model
