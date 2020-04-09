import numpy as np
from keras import layers, models, optimizers, regularizers
from keras import backend as K

from capsNet_layers import CapsuleLayer, PrimaryCaps, Length

# The Model proposed by J. Kim et al.
def CapsNet(input_shape, n_classes, num_routings, vocab_size, embedding_dim, l2_ratio):
	'''
	A Capsule Network for Text Classification.
	: param input_shape : data shape, 3d, [width, height, channels].
	: param n_classes : number of class.
	: param num_routings : number of routing iterations.
	'''
	input_seq = layers.Input(shape=input_shape)
	embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
		embeddings_regularizer=regularizers.l2(l2_ratio), mask_zero=True, trainable=True)(input_seq)
	embeddings = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(embeddings) # Reshape sequence for convolution.

	# Non-linear gate layer.
	elu_layer = layers.Conv2D(filters=256, kernel_size=(3, embedding_dim), strides=1,
		use_bias=False, kernel_regularizer=regularizers.l2(l2_ratio), activation=None)(embeddings)
	elu_layer = layers.BatchNormalization()(elu_layer)
	elu_layer = layers.Activation('elu')(elu_layer)

	conv_layer = layers.Conv2D(filters=256, kernel_size=(3, embedding_dim), strides=1,
		use_bias=False, kernel_regularizer=regularizers.l2(l2_ratio), activation=None)(embeddings)
	conv_layer = layers.BatchNormalization()(conv_layer)

	gate_layer = layers.Multiply()([elu_layer, conv_layer])

	# Using Dropout.
	gate_layer = layers.Dropout(0.1)(gate_layer)

	# Capsule layer.
	# primary_caps = layers.Conv2D(filters=num_capsule*16, kernel_size=(K.int_shape(gate_layer)[1], 1),
	# 	use_bias=False, kernel_regularizer=regularizers.l2(l2_ratio), activation=None)(gate_layer)
	primary_caps = layers.Conv2D(filters=6*16, kernel_size=(K.int_shape(gate_layer)[1], 1),
		strides=1, use_bias=False, kernel_regularizer=regularizers.l2(l2_ratio), activation=None)(gate_layer)
	primary_caps = layers.Reshape((6, 16))(primary_caps)
	primary_caps = layers.BatchNormalization()(primary_caps)
	primary_caps = layers.Activation('relu')(primary_caps)

	# Using Dropout.
	primary_caps = layers.Dropout(0.1)(primary_caps)

	text_caps = CapsuleLayer(num_capsule=n_classes, dim_capsule=16,
		num_routings=num_routings, name='text_caps')(primary_caps)

	outputs = Length(name='L2_norm')(text_caps)

	model = models.Model(inputs=input_seq, outputs=outputs)
	print(model.summary())

	return model

# # Original capsule networks (Hinton).
# def CapsNet(input_shape, n_classes, num_routings, vocab_size, embedding_dim, l2_ratio, max_len):
# 	"""
# 	:param input_shape: data shape, 4d, [None, width, height, channels]
# 	:param n_classes: number of classes
# 	:param num_routing: number of routing iterations
# 	:param vocab_size:
# 	:param embed_dim:
# 	:param max_len:
# 	:return: A Keras Model with 2 inputs and 2 outputs
# 	"""
# 	print(input_shape)
# 	x = layers.Input(shape=input_shape)
# 	embed = layers.Embedding(vocab_size, embedding_dim, input_length=max_len)(x)
# 	embed = layers.Lambda(lambda x: K.expand_dims(x, axis=-1))(embed)
# 	print('Embedding: ', embed.shape)

# 	# Layer 1: Conv1D layer
# 	conv = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001), name='conv1')(embed)
# 	print('Conv1: ', conv.shape)

# 	# # Layer 2: Dropout regularization
# 	# dropout = layers.Dropout(0.5)(conv)

# 	# Layer 3: Primary layer with `squash` activation, then reshape to [None, num_capsule, dim_vector]
# 	primary_caps = PrimaryCaps(conv, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
# 	print('primary_caps: ', primary_caps.shape)
# 	print()
# 	print()

# 	# Layer 4: Capsule layer. Routing algorithm works here.
# 	category_caps = CapsuleLayer(num_capsule=n_classes, dim_capsule=16, num_routings=num_routings, name='category_caps')(primary_caps)

# 	# Layer 5: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
# 	out_caps = Length(name='out_caps')(category_caps)
# 	# out_caps = layers.Lambda(lambda x: K.l2_normalize(x, axis=1))(out_caps)

# 	return models.Model(input=x, output=out_caps)
