# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import backend as K

##### Custom layers for Capsule Networks.
'''
Implementation of Capsule block (or layer), CapsuleNorm and Squach function,
which was first proposed in "Dynamic Routing Between Capsules" (Sabour et al., 2017).
[https://proceedings.neurips.cc/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html]

- Reference: https://github.com/khikmatullaev/CapsNet-Keras-Text-Classification
'''

def squash_fn(vectors, axis=-1):
	s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
	scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm)
	return scale * vectors

class CapsuleNorm(tf.keras.layers.Layer):
	'''
	Compute the length of vectors. This is used to compute a Tensor that has the same shape with class in margin_loss.
	'''
	def call(self, inputs, **kwargs):
		return K.sqrt(K.sum(K.square(inputs), -1) + K.epsilon())

	def compute_output_shape(self, input_shape):
		return input_shape[:-1]

class CapsuleBlock(tf.keras.layers.Layer):
	def __init__(self, num_capsule, dim_capsule, routing, num_routings, **kwargs):
		super(CapsuleBlock, self).__init__(**kwargs)
		self.num_capsule = num_capsule
		self.dim_capsule = dim_capsule
		self.routing = routing
		self.num_routings = num_routings
		self.kernel_initializer = tf.keras.initializers.get('glorot_uniform')

	def build(self, input_shape):
		# input_shape == [None, input_num_capsule, input_dim_capsule]
		self.input_num_capsule = input_shape[1]
		self.input_dim_capsule = input_shape[-1]

		self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, self.input_dim_capsule, self.dim_capsule], initializer=self.kernel_initializer, trainable=True, name='W')

		self.built = True

	def call(self, inputs, training=True):
		# inputs.shape == [None, input_num_capsule, input_dim_capsule]
		# expand.shape == [None, input_num_capsule, 1, 1, input_dim_capsule]
		expand = K.expand_dims(K.expand_dims(inputs, 2), 2)
		# tiled.shape == [None, input_num_capsule, num_capsule, 1, input_dim_capsule]
		tiled = K.tile(expand, [1, 1, self.num_capsule, 1, 1])
		# hats.shape == [None, input_num_capsule, num_capsule, 1, dim_capsule]
		hats = K.map_fn(lambda x: tf.matmul(x, self.W), elems=tiled)

		#
		if self.routing:
			print('DYNAMIC ROUTING..!')
			# bias.shape == [1, input_num_capsue, num_capsule, 1, 1]
			bias = tf.zeros(shape=[1, self.input_num_capsule, self.num_capsule, 1, 1])
			for i in range(self.num_routings):
				c = tf.nn.softmax(bias, axis=2)
				# output.shape == [None, 1, num_capsule, 1, dim_capsule]
				output = squash_fn(tf.compat.v1.reduce_sum(c * hats, axis=1, keep_dims=True))

				if i < (self.num_routings - 1):
					bias += tf.compat.v1.reduce_sum(hats * output, axis=-1, keep_dims=True)
		else:
			print('STATIC ROUTING..!')
			output = K.sum(hats, axis=2)
			output = squash_fn(output)

		# output.shape == [None, num_capsule, dim_capsule]
		return tf.reshape(output, [-1, self.num_capsule, self.dim_capsule])

	def compute_output_shape(self, input_shape):
		return tuple([None, self.num_capsule, self.dim_capsule])

'''
def PrimaryCaps(inputs, dim_capsule, n_channels, kernel_size, strides, padding, name):
	output = tf.keras.layers.Conv1D(filters=(dim_capsule * n_channels), kernel_size=kernel_size, strides=strides, padding=padding, name=name)(inputs)
	output = tf.keras.layers.Reshape(target_shape=[-1, dim_capsule])(output)
	return tf.keras.layers.Lambda(squash_fn)(output)
'''