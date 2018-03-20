# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils
import numpy as np
from tensorflow import flags
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers

FLAGS = flags.FLAGS
#flags.DEFINE_integer("hidden_size", 8192, "NN hidden size")
flags.DEFINE_float("sharpening", 1.0, "Corelation matrix sharpening parameters")
flags.DEFINE_float("corelation_gamma", 0.1, "corelation matrix strength")
flags.DEFINE_integer(
	"moe_num_mixtures", 2,
	"The number of mixtures (excluding the dummy 'expert') used for MoeModel.")

flags.DEFINE_integer(
	"dim_gate", 2048,
	"2-layer moe model dimension")

flags.DEFINE_integer(
	"bottleneck_size", 2048,
	"bottleneck feature dimension")


class LogisticModel(models.BaseModel):
	"""Logistic model with L2 regularization."""

	def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
		"""Creates a logistic model.
		Args:
		  model_input: 'batch' x 'num_features' matrix of input features.
		  vocab_size: The number of classes in the dataset.
		Returns:
		  A dictionary with a tensor containing the probability predictions of the
		  model in the 'predictions' key. The dimensions of the tensor are
		  batch_size x num_classes."""
		output = slim.fully_connected(
			model_input, vocab_size, activation_fn=tf.nn.sigmoid,
			weights_regularizer=slim.l2_regularizer(l2_penalty))
		return {"predictions": output}


class Logistic_Multi_Layer_Model(models.BaseModel):
	"""Logistic model with L2 regularization."""

	def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
		"""Creates a Multi-layer bottleneck logistic model.
		Args:
		  model_input: 'batch' x 'num_features' matrix of input features.
		  vocab_size: The number of classes in the dataset.
		Returns:
		  A dictionary with a tensor containing the probability predictions of the
		  model in the 'predictions' key. The dimensions of the tensor are
		  batch_size x num_classes."""

		# hidden_size = FLAGS.hidden_size
		# bottleneck_size = FLAGS.bottleneck_size

		hidden_size = 2048*4
		bottleneck_size = vocab_size
		

		fc1 = slim.fully_connected(
			model_input, hidden_size, activation_fn=None, biases_initializer=None,
			weights_regularizer=slim.l2_regularizer(l2_penalty))
		fc1 = tf.contrib.layers.layer_norm(inputs=fc1, center=True, scale=True, activation_fn=tf.nn.relu)

		fc2 = slim.fully_connected(
			fc1, bottleneck_size, activation_fn=None, biases_initializer=None,
			weights_regularizer=slim.l2_regularizer(l2_penalty))
		fc2 = tf.contrib.layers.layer_norm(inputs=fc2, center=True, scale=True, activation_fn=tf.nn.relu)

		output = slim.fully_connected(
			fc2, vocab_size, activation_fn=tf.nn.sigmoid,
			weights_regularizer=slim.l2_regularizer(l2_penalty))
		return {"predictions": output, "features": fc2}


class NN(models.BaseModel):
	"""A softmax over a mixture of logistic models (with L2 regularization)."""

	def create_model(self,
	                 model_input,
	                 vocab_size,
	                 num_mixtures=None,
	                 l2_penalty=1e-8,
	                 **unused_params):
		hidden_size = 8193
		h1 = model_input
		num_layer = 3

		for _ in range(num_layer):
			h1 = slim.fully_connected(
				h1,
				hidden_size,
				activation_fn=None,
				biases_initializer=None,
				weights_regularizer=slim.l2_regularizer(l2_penalty))

			h1 = tf.contrib.layers.layer_norm(inputs=h1, center=True, scale=True, activation_fn=tf.nn.relu)

		predictions = tf.nn.sigmoid(
			slim.fully_connected(
				h1,
				vocab_size,
				activation_fn=None,
				biases_initializer=None,
				weights_regularizer=slim.l2_regularizer(l2_penalty),
				scope="predictions_layer")
		)

		return {"predictions": predictions}


class MoeModel(models.BaseModel):
	"""A softmax over a mixture of logistic models (with L2 regularization)."""

	def create_model(self,
	                 model_input,
	                 vocab_size,
	                 num_mixtures=None,
	                 l2_penalty=1e-8,
	                 **unused_params):
		"""Creates a Mixture of (Logistic) Experts model.
		 The model consists of a per-class softmax distribution over a
		 configurable number of logistic classifiers. One of the classifiers in the
		 mixture is not trained, and always predicts 0.
		Args:
		  model_input: 'batch_size' x 'num_features' matrix of input features.
		  vocab_size: The number of classes in the dataset.
		  num_mixtures: The number of mixtures (excluding a dummy 'expert' that
			always predicts the non-existence of an entity).
		  l2_penalty: How much to penalize the squared magnitudes of parameter
			values.
		Returns:
		  A dictionary with a tensor containing the probability predictions of the
		  model in the 'predictions' key. The dimensions of the tensor are
		  batch_size x num_classes.
		"""
		num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

		gate_activations = slim.fully_connected(
			model_input,
			vocab_size * (num_mixtures + 1),
			activation_fn=None,
			biases_initializer=None,
			weights_regularizer=slim.l2_regularizer(l2_penalty),
			scope="gates")

		expert_activations = slim.fully_connected(
			model_input,
			vocab_size * num_mixtures,
			activation_fn=None,
			weights_regularizer=slim.l2_regularizer(l2_penalty),
			scope="experts")

		gating_distribution = tf.nn.softmax(tf.reshape(
			gate_activations,
			[-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
		expert_distribution = tf.nn.sigmoid(tf.reshape(
			expert_activations,
			[-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

		final_probabilities_by_class_and_batch = tf.reduce_sum(
			gating_distribution[:, :num_mixtures] * expert_distribution, 1)
		final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
		                                 [-1, vocab_size])
		return {"predictions": final_probabilities}


class Moe_2Layer_Model(models.BaseModel):
	"""A softmax over a mixture of logistic models (with L2 regularization)."""

	def create_model(self,
	                 model_input,
	                 vocab_size,
	                 num_mixtures=None,
	                 l2_penalty=1e-8,
	                 **unused_params):
		"""Creates a Mixture of (Logistic) Experts model.
		The model consists of a per-class softmax distribution over a
		configurable number of logistic classifiers. One of the classifiers in the
		mixture is not trained, and always predicts 0.
		Args:
		  model_input: 'batch_size' x 'num_features' matrix of input features.
		  vocab_size: The number of classes in the dataset.
		  num_mixtures: The number of mixtures (excluding a dummy 'expert' that
			always predicts the non-existence of an entity).
		  l2_penalty: How much to penalize the squared magnitudes of parameter
			values.
		Returns:
		  A dictionary with a tensor containing the probability predictions of the
		  model in the 'predictions' key. The dimensions of the tensor are
		  batch_size x num_classes.
		"""
		num_mixtures = num_mixtures or FLAGS.moe_num_mixtures
		dim_gate = FLAGS.dim_gate

		gate_activations = slim.fully_connected(
			model_input,
			dim_gate,
			activation_fn=None,
			weights_regularizer=slim.l2_regularizer(l2_penalty),
			scope="gates0")

		gate_activations = slim.fully_connected(
			gate_activations,
			vocab_size * (num_mixtures + 1),
			activation_fn=None,
			biases_initializer=None,
			weights_regularizer=slim.l2_regularizer(l2_penalty),
			scope="gates")
		expert_activations = slim.fully_connected(
			model_input,
			vocab_size * num_mixtures,
			activation_fn=None,
			weights_regularizer=slim.l2_regularizer(l2_penalty),
			scope="experts")

		gating_distribution = tf.nn.softmax(tf.reshape(
			gate_activations,
			[-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
		expert_distribution = tf.nn.sigmoid(tf.reshape(
			expert_activations,
			[-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

		final_probabilities_by_class_and_batch = tf.reduce_sum(
			gating_distribution[:, :num_mixtures] * expert_distribution, 1)
		final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
		                                 [-1, vocab_size])
		return {"predictions": final_probabilities}


class MoNN3Lw(models.BaseModel):
	"""A softmax over a mixture of logistic models (with L2 regularization)."""

	def create_model(self,
	                 model_input,
	                 vocab_size,
	                 num_mixtures=None,
	                 l2_penalty=1e-8,
	                 **unused_params):
		num_mixtures = 3

		gate_activations = slim.fully_connected(
			model_input,
			vocab_size * (num_mixtures + 1),
			activation_fn=None,
			biases_initializer=None,
			weights_regularizer=slim.l2_regularizer(l2_penalty),
			scope="gates")

		a1Units = 8193 * 3
		A1 = slim.fully_connected(
			model_input, a1Units, activation_fn=tf.nn.relu,
			weights_regularizer=slim.l2_regularizer(l2_penalty),
			scope='FC_HA1')
		a2Units = 8193
		A2 = slim.fully_connected(
			A1, a2Units, activation_fn=tf.nn.relu,
			weights_regularizer=slim.l2_regularizer(l2_penalty),
			scope='FC_HA2')
		a2Units = 8193 * 2
		A3 = slim.fully_connected(
			A2, a2Units, activation_fn=tf.nn.relu,
			weights_regularizer=slim.l2_regularizer(l2_penalty),
			scope='FC_HA3')

		expert_activations = slim.fully_connected(
			A3,
			vocab_size * num_mixtures,
			activation_fn=None,
			weights_regularizer=slim.l2_regularizer(l2_penalty),
			scope="experts")

		gating_distribution = tf.nn.softmax(tf.reshape(
			gate_activations,
			[-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
		expert_distribution = tf.nn.sigmoid(tf.reshape(
			expert_activations,
			[-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

		final_probabilities_by_class_and_batch = tf.reduce_sum(
			gating_distribution[:, :num_mixtures] * expert_distribution, 1)
		final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
		                                 [-1, vocab_size])
		return {"predictions": final_probabilities}


