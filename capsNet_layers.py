import tensorflow as tf

from keras import backend as K
from keras import layers, models, optimizers, initializers, regularizers


def squash(vectors, axis=-1):
	squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
	return vectors * (squared_norm / ((1 + squared_norm) * tf.sqrt(squared_norm + K.epsilon())))


class Length(layers.Layer):
	def call(self, inputs, **kwargs):
		return tf.sqrt(tf.reduce_sum(tf.square(inputs), axis=-1, keep_dims=False) + K.epsilon())

	def compute_output_shape(self, input_shape):
		*output_shape, _ = input_shape
		return tuple(output_shape)


class Mask(layers.Layer):
	def call(self, inputs, **kwargs):
		inputs, mask = inputs
		return K.batch_dot(inputs, mask, 1)

	def compute_output_shape(self, input_shape):
		*_, output_shape = input_shape[0]
		return (None, output_shape)


class CapsuleLayer(layers.Layer):
	def __init__(self, num_capsule, dim_capsule, num_routings, **kwargs):
		super(CapsuleLayer, self).__init__(**kwargs)
		self.num_capsule = num_capsule
		self.dim_capsule = dim_capsule
		self.num_routings = num_routings
		self.kernel_initializer = initializers.get('he_normal')
		self.bias_initializer = initializers.get('zeros')

	def build(self, input_shape):
		# 'input_shape' is a 4D tensor.
		self.input_num_capsule = input_shape[1]
		self.input_dim_capsule = input_shape[2]
		# _, self.input_num_capsule, self.input_dim_capsule, *_ = input_shape
		self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, self.input_dim_capsule, self.dim_capsule], initializer=self.kernel_initializer, name='W')
		self.bias = self.add_weight(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1], initializer=self.bias_initializer, name='bias', trainable=False)

		self.built = True

	def call(self, inputs, training=None):
		print('---- inputs : ', inputs.shape, ' ----')

		inputs_expand = tf.expand_dims(tf.expand_dims(inputs, 2), 2)
		print('---- inputs_expand : ', inputs_expand.shape, ' ----')

		inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsule, 1, 1])
		print('---- inputs_tiled : ', inputs_tiled.shape, ' ----')
		print('---- self.W : ', self.W.shape, ' ----')

		# inputs_hat = tf.scan(lambda z, x: K.batch_dot(x, self.W, [3, 2]), elems=inputs_tiled,
		# 	initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_capsule]))
		inputs_hat = tf.scan(lambda z, x: tf.matmul(x, self.W), elems=inputs_tiled,
			initializer=K.zeros([self.input_num_capsule, self.num_capsule, 1, self.dim_capsule]))
		print('---- inputs_hat : ', inputs_hat.shape, ' ----')
		print('---- bias : ', self.bias.shape, ' ----')

		for i in range(self.num_routings):
			c = tf.nn.softmax(self.bias, dim=2)
			outputs = squash(tf.reduce_sum(c * inputs_hat, axis=1, keep_dims=True))

			if i < self.num_routings-1:
				# self.bias += tf.reduce_sum(inputs_hat * outputs, axis=-1, keep_dims=True)
				self.bias.assign_add(tf.reduce_sum(inputs_hat * outputs, axis=-1, keep_dims=True))

		return tf.reshape(outputs, [-1, self.num_capsule, self.dim_capsule])

	def compute_output_shape(self, input_shape):
		return (None, self.num_capsule, self.dim_capsule)


def PrimaryCaps(inputs, dim_capsule, n_channels, kernel_size, strides, padding='valid'):
	outputs = layers.Conv2D(filters=dim_capsule*n_channels, kernel_size=kernel_size,
		strides=strides, padding=padding)(inputs)
	outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primary_caps_reshape')(outputs)
	return layers.Lambda(squash, name='primary_caps_squash')(outputs)