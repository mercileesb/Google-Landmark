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

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import numpy as np
import models
import video_level_models
import tensorflow as tf
import model_utils as utils

import tensorflow.contrib.slim as slim
from tensorflow import flags

FLAGS = flags.FLAGS


"""
Willow
"""

flags.DEFINE_bool("gating_remove_diag", False,
                  "Remove diag for self gating")
flags.DEFINE_bool("lightvlad", False,
                  "Light or full NetVLAD")
flags.DEFINE_bool("vlagd", False,
                  "vlagd of vlad")

flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
	"sample_random_frames", True,
	"If true samples random frames (for frame level models). If false, a random"
	"sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 16384,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 2048,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_bool("dbof_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("dbof_var_features", 0,
                     "Variance features on top of Dbof cluster layer.")

flags.DEFINE_string("dbof_activation", "relu", 'dbof activation')

flags.DEFINE_bool("softdbof_maxpool", False, 'add max pool to soft dbof')

flags.DEFINE_integer("netvlad_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")
flags.DEFINE_bool("netvlad_relu", True, 'add ReLU to hidden layer')
flags.DEFINE_integer("netvlad_dimred", -1,
                     "NetVLAD output dimension reduction")
flags.DEFINE_integer("gatednetvlad_dimred", 1024,
                     "GatedNetVLAD output dimension reduction")

flags.DEFINE_bool("gating", False,
                  "Gating for NetVLAD")
flags.DEFINE_integer("hidden_size", 1024,
                     "size of hidden layer for BasicStatModel.")

flags.DEFINE_integer("netvlad_hidden_size", 1024,
                     "Number of units in the NetVLAD hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_video", 1024,
                     "Number of units in the NetVLAD video hidden layer.")

flags.DEFINE_integer("netvlad_hidden_size_audio", 64,
                     "Number of units in the NetVLAD audio hidden layer.")

flags.DEFINE_bool("netvlad_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")

flags.DEFINE_integer("fv_cluster_size", 64,
                     "Number of units in the NetVLAD cluster layer.")

flags.DEFINE_integer("fv_hidden_size", 2048,
                     "Number of units in the NetVLAD hidden layer.")
flags.DEFINE_bool("fv_relu", True,
                  "ReLU after the NetFV hidden layer.")

flags.DEFINE_bool("fv_couple_weights", True,
                  "Coupling cluster weights or not")

flags.DEFINE_integer("fv_coupling_factor", 0.01,
                     "Coupling factor")

flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_integer("lstm_cells_video", 1024, "Number of LSTM cells (video).")
flags.DEFINE_integer("lstm_cells_audio", 128, "Number of LSTM cells (audio).")

flags.DEFINE_integer("gru_cells", 1024, "Number of GRU cells.")
flags.DEFINE_integer("gru_cells_video", 1024, "Number of GRU cells (video).")
flags.DEFINE_integer("gru_cells_audio", 128, "Number of GRU cells (audio).")
flags.DEFINE_integer("gru_layers", 2, "Number of GRU layers.")
flags.DEFINE_bool("lstm_random_sequence", False,
                  "Random sequence input for lstm.")
flags.DEFINE_bool("gru_random_sequence", False,
                  "Random sequence input for gru.")
flags.DEFINE_bool("gru_backward", False, "BW reading for GRU")
flags.DEFINE_bool("lstm_backward", False, "BW reading for LSTM")

flags.DEFINE_bool("fc_dimred", True, "Adding FC dimred after pooling")


class FrameLevelLogisticModel(models.BaseModel):
	def create_model(self, model_input, vocab_size, num_frames, **unused_params):
		"""Creates a model which uses a logistic classifier over the average of the
		frame-level features.
		This class is intended to be an example for implementors of frame level
		models. If you want to train a model over averaged features it is more
		efficient to average them beforehand rather than on the fly.
		Args:
		  model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
					   input features.
		  vocab_size: The number of classes in the dataset.
		  num_frames: A vector of length 'batch' which indicates the number of
			   frames for each video (before padding).
		Returns:
		  A dictionary with a tensor containing the probability predictions of the
		  model in the 'predictions' key. The dimensions of the tensor are
		  'batch_size' x 'num_classes'.
		"""
		num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
		feature_size = model_input.get_shape().as_list()[2]

		denominators = tf.reshape(
			tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
		avg_pooled = tf.reduce_sum(model_input,
		                           axis=[1]) / denominators

		output = slim.fully_connected(
			avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
			weights_regularizer=slim.l2_regularizer(1e-8))
		return {"predictions": output}


class DbofModel(models.BaseModel):
	"""Creates a Deep Bag of Frames model.
	The model projects the features for each frame into a higher dimensional
	'clustering' space, pools across frames in that space, and then
	uses a configurable video-level model to classify the now aggregated features.
	The model will randomly sample either frames or sequences of frames during
	training to speed up convergence.
	Args:
	  model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
				   input features.
	  vocab_size: The number of classes in the dataset.
	  num_frames: A vector of length 'batch' which indicates the number of
		   frames for each video (before padding).
	Returns:
	  A dictionary with a tensor containing the probability predictions of the
	  model in the 'predictions' key. The dimensions of the tensor are
	  'batch_size' x 'num_classes'.
	"""

	def create_model(self,
	                 model_input,
	                 vocab_size,
	                 num_frames,
	                 iterations=None,
	                 add_batch_norm=None,
	                 sample_random_frames=None,
	                 cluster_size=None,
	                 hidden_size=None,
	                 is_training=True,
	                 **unused_params):
		iterations = iterations or FLAGS.iterations
		add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
		random_frames = sample_random_frames or FLAGS.sample_random_frames
		cluster_size = cluster_size or FLAGS.dbof_cluster_size
		hidden1_size = hidden_size or FLAGS.dbof_hidden_size

		num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
		if random_frames:
			model_input = utils.SampleRandomFrames(model_input, num_frames,
			                                       iterations)
		else:
			model_input = utils.SampleRandomSequence(model_input, num_frames,
			                                         iterations)
		max_frames = model_input.get_shape().as_list()[1]
		feature_size = model_input.get_shape().as_list()[2]
		reshaped_input = tf.reshape(model_input, [-1, feature_size])
		tf.summary.histogram("input_hist", reshaped_input)

		if add_batch_norm:
			reshaped_input = slim.batch_norm(
				reshaped_input,
				center=True,
				scale=True,
				is_training=is_training,
				scope="input_bn")

		cluster_weights = tf.get_variable("cluster_weights",
		                                  [feature_size, cluster_size],
		                                  initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
		tf.summary.histogram("cluster_weights", cluster_weights)
		activation = tf.matmul(reshaped_input, cluster_weights)
		if add_batch_norm:
			activation = slim.batch_norm(
				activation,
				center=True,
				scale=True,
				is_training=is_training,
				scope="cluster_bn")
		else:
			cluster_biases = tf.get_variable("cluster_biases",
			                                 [cluster_size],
			                                 initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
			tf.summary.histogram("cluster_biases", cluster_biases)
			activation += cluster_biases
		activation = tf.nn.relu6(activation)
		tf.summary.histogram("cluster_output", activation)

		activation = tf.reshape(activation, [-1, max_frames, cluster_size])
		activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

		hidden1_weights = tf.get_variable("hidden1_weights",
		                                  [cluster_size, hidden1_size],
		                                  initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
		tf.summary.histogram("hidden1_weights", hidden1_weights)
		activation = tf.matmul(activation, hidden1_weights)
		if add_batch_norm:
			activation = slim.batch_norm(
				activation,
				center=True,
				scale=True,
				is_training=is_training,
				scope="hidden1_bn")
		else:
			hidden1_biases = tf.get_variable("hidden1_biases",
			                                 [hidden1_size],
			                                 initializer=tf.random_normal_initializer(stddev=0.01))
			tf.summary.histogram("hidden1_biases", hidden1_biases)
			activation += hidden1_biases
		activation = tf.nn.relu6(activation)
		tf.summary.histogram("hidden1_output", activation)

		aggregated_model = getattr(video_level_models,
		                           FLAGS.video_level_classifier_model)
		return aggregated_model().create_model(
			model_input=activation,
			vocab_size=vocab_size,
			**unused_params)


class LayerNormLstmAveConcatModel(models.BaseModel):
	def create_model(self, model_input, vocab_size, num_frames, **unused_params):
		lstm_size = FLAGS.lstm_cells
		number_of_layers = FLAGS.lstm_layers

		stacked_lstm = tf.contrib.rnn.LayerNormBasicLSTMCell(
			num_units=lstm_size,
			dropout_keep_prob=0.5)

		loss = 0.0

		model_input = slim.fully_connected(
			model_input, 1024, activation_fn=tf.nn.sigmoid,
			weights_regularizer=slim.l2_regularizer(1e-8))

		outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
		                                   sequence_length=num_frames,
		                                   dtype=tf.float32)

		state = tf.concat([state[0], state[1]], 1)

		average_state = tf.nn.l2_normalize(tf.reduce_sum(model_input, axis=1), dim=1)
		state = tf.concat([state, average_state], 1)

		aggregated_model = getattr(video_level_models,
		                           FLAGS.video_level_classifier_model)

		return aggregated_model().create_model(
			model_input=state,
			vocab_size=vocab_size,
			**unused_params)


class LstmModel(models.BaseModel):
	def create_model(self, model_input, vocab_size, num_frames, **unused_params):
		"""Creates a model which uses a stack of LSTMs to represent the video.
		Args:
		  model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
					   input features.
		  vocab_size: The number of classes in the dataset.
		  num_frames: A vector of length 'batch' which indicates the number of
			   frames for each video (before padding).
		Returns:
		  A dictionary with a tensor containing the probability predictions of the
		  model in the 'predictions' key. The dimensions of the tensor are
		  'batch_size' x 'num_classes'.
		"""
		lstm_size = FLAGS.lstm_cells
		number_of_layers = FLAGS.lstm_layers

		stacked_lstm = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.BasicLSTMCell(
					lstm_size, forget_bias=1.0, state_is_tuple=False)
				for _ in range(number_of_layers)
			], state_is_tuple=False)

		loss = 0.0

		model_input = slim.fully_connected(
			model_input, 1024, activation_fn=tf.nn.sigmoid,
			weights_regularizer=slim.l2_regularizer(1e-8))


		outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
		                                   sequence_length=num_frames,
		                                   swap_memory=True,
		                                   dtype=tf.float32)

		aggregated_model = getattr(video_level_models,
		                           FLAGS.video_level_classifier_model)

		return aggregated_model().create_model(
			model_input=state,
			vocab_size=vocab_size,
			**unused_params)


class ContextMemoryModel(models.BaseModel):
	def create_model(self, model_input, vocab_size, num_frames, **unused_params):
		lstm_size = FLAGS.lstm_cells
		number_of_layers = 1

		stacked_lstm = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.BasicLSTMCell(
					lstm_size, forget_bias=1.0, state_is_tuple=False,
					)
				for _ in range(number_of_layers)
			], state_is_tuple=False)

		loss = 0.0

		outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
		                                   sequence_length=num_frames,
		                                   swap_memory=True,
		                                   dtype=tf.float32)

		context_memory = tf.nn.l2_normalize(tf.reduce_sum(outputs, axis=1), dim=1)
		average_state = tf.nn.l2_normalize(tf.reduce_sum(model_input, axis=1), dim=1)
		# state = tf.concat([state[0], state[1]], 1)

		final_state = tf.concat([context_memory, state, average_state], 1)

		aggregated_model = getattr(video_level_models,
		                           FLAGS.video_level_classifier_model)

		return aggregated_model().create_model(
			model_input=final_state,
			vocab_size=vocab_size,
			**unused_params)


class Many2ManyLstmModel(models.BaseModel):
	def create_model(self, model_input, vocab_size, num_frames, **unused_params):
		lstm_size = FLAGS.lstm_cells
		number_of_layers = FLAGS.lstm_layers

		cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
		outputs, state = tf.nn.dynamic_rnn(
			cell=cell,
			inputs=model_input,
			sequence_length=num_frames,
			parallel_iterations=128,
			dtype=tf.float32)  # output = (batch, num_frames, lstm_size)

		class_per_output = slim.fully_connected(
			outputs,
			4716,
			activation_fn=tf.nn.sigmoid,
			weights_regularizer=slim.l2_regularizer(1e-8))  # (batch, num_frames, 4716)

		final_probabilities = tf.reduce_mean(class_per_output, 1)
		return {"predictions": final_probabilities}


class PositionEncodingModel(models.BaseModel):
	def create_model(self, model_input, vocab_size, num_frames, **unused_params):
		"""Creates a model which uses a stack of LSTMs to represent the video.
		Args:
		  model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
					   input features.
		  vocab_size: The number of classes in the dataset.
		  num_frames: A vector of length 'batch' which indicates the number of
			   frames for each video (before padding).
		Returns:
		  A dictionary with a tensor containing the probability predictions of the
		  model in the 'predictions' key. The dimensions of the tensor are
		  'batch_size' x 'num_classes'.
		"""

		J = 300
		d = FLAGS.feature_dim

		# PE matrix
		l = [[(1 - j / J) - (k / d) * (1 - 2 * j / J) for k in range(d)] for j in range(J)]

		# Adding Gaussian Noise
		state = model_input + FLAGS.gaussian_noise * tf.random_normal(shape=[J, d])
		state = tf.reduce_sum(state * l, axis=1)
		state = tf.nn.l2_normalize(state, dim=1, epsilon=1e-6)
		aggregated_model = getattr(video_level_models,
		                           FLAGS.video_level_classifier_model)

		return aggregated_model().create_model(
			model_input=state,
			vocab_size=vocab_size,
			**unused_params)


class BiLstmModel(models.BaseModel):
	def create_model(self, model_input, vocab_size, num_frames, **unused_params):
		"""Creates a model which uses a stack of LSTMs to represent the video.
		Args:
		  model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
					   input features.
		  vocab_size: The number of classes in the dataset.
		  num_frames: A vector of length 'batch' which indicates the number of
			   frames for each video (before padding).
		Returns:
		  A dictionary with a tensor containing the probability predictions of the
		  model in the 'predictions' key. The dimensions of the tensor are
		  'batch_size' x 'num_classes'.
		"""
		lstm_size = FLAGS.lstm_cells
		number_of_layers = FLAGS.lstm_layers

		model_input = slim.fully_connected(
			model_input, 1024, activation_fn=tf.nn.sigmoid,
			weights_regularizer=slim.l2_regularizer(1e-8))

		stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.BasicLSTMCell(
					lstm_size, forget_bias=1.0, state_is_tuple=False)
				for _ in range(number_of_layers)
			], state_is_tuple=False)

		stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.BasicLSTMCell(
					lstm_size, forget_bias=1.0, state_is_tuple=False)
				for _ in range(number_of_layers)
			], state_is_tuple=False)

		loss = 0.0

		output, state = tf.nn.bidirectional_dynamic_rnn(
			cell_fw=stacked_lstm_fw,
			cell_bw=stacked_lstm_bw,
			inputs=model_input,
			sequence_length=num_frames,
			dtype=tf.float32)

		state = tf.concat([state[0], state[1]], 1)
		aggregated_model = getattr(video_level_models,
		                           FLAGS.video_level_classifier_model)

		return aggregated_model().create_model(
			model_input=state,
			vocab_size=vocab_size,
			**unused_params)


class ordinalTopK(models.BaseModel):
	def create_model(self, model_input, vocab_size, num_frames, **unused_params):
		"""Creates a model which uses a ordinal TopK to represent the video.
		Args:
		model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
					  input features.
		vocab_size: The number of classes in the dataset.
		num_frames: A vector of length 'batch' which indicates the number of
		  frames for each video (before padding).
		Returns:
		  A dictionary with a tensor containing the probability predictions of the
		  model in the 'predictions' key. The dimensions of the tensor are
		  'batch_size' x 'num_classes'.
		"""

		num_features = FLAGS.feature_dim
		k = FLAGS.topk
		mean, var = tf.nn.moments(x=model_input, axes=[1], keep_dims=True)  # [batch_size, 1, num_featuers]
		model_input_trans = tf.transpose(a=model_input, perm=[0, 2, 1])  # [batch_size, num_features, max_frames]
		topk, indices = tf.nn.top_k(input=model_input_trans, k=k, sorted=True)  # topk: [batch_size, num_features, k]
		topk_trans = tf.transpose(a=topk, perm=[0, 2, 1])  # [batch_size, k, num_features]
		topk_trans = tf.nn.l2_normalize(topk_trans, dim=2)

		concat = tf.concat([mean, var, topk_trans], 1)  # [batch_size, k+2, num_features]
		concat_flat = tf.reshape(concat, shape=[-1, (k + 2) * num_features])  # [batch_size, (k+2) * num_featuers]
		# concat_flat = tf.nn.l2_normalize(concat_flat, dim=1)
		concat_flat.set_shape([None, (k + 2) * num_features])

		aggregated_model = getattr(video_level_models, FLAGS.video_level_classifier_model)

		return aggregated_model().create_model(
			model_input=concat_flat,
			vocab_size=vocab_size,
			**unused_params)


class CNN(models.BaseModel):
	def create_model(self, model_input, vocab_size, num_frames, **unused_params):
		max_frames = 300
		model_input = tf.expand_dims(model_input, -1)  # [batch_size, max_frames, num_features, 1]
		num_channels = FLAGS.num_channels

		kernel_height = FLAGS.kernel_height
		kernel_width = FLAGS.kernel_width

		# [batch_size, max_frames - k, 1, num_channels]
		cnn_activation = tf.contrib.layers.conv2d(
			inputs=model_input,
			num_outputs=num_channels,
			kernel_size=[kernel_height, kernel_width],
			stride=1,
			padding="VALID",
			activation_fn=tf.nn.relu,
			trainable=True
		)

		# cnn_activation = tf.nn.l2_normalize(cnn_activation, dim=3)

		# [batch_size, 1, 1, num_channels]
		max_pool_over_time = tf.contrib.layers.max_pool2d(
			inputs=cnn_activation,
			kernel_size=[max_frames - kernel_height + 1, 1],
			stride=1,
			padding="VALID")

		state = tf.reshape(max_pool_over_time, shape=[-1, num_channels])
		state = tf.nn.l2_normalize(state, dim=1)

		aggregated_model = getattr(video_level_models, FLAGS.video_level_classifier_model)
		return aggregated_model().create_model(
			model_input=state,
			vocab_size=vocab_size,
			**unused_params)


class Lstm_average_concat(models.BaseModel):
	def create_model(self, model_input, vocab_size, num_frames, **unused_params):
		if FLAGS.position_encoding == True:
			J = 300
			d = 1152
			l = [[(1 - j / J) - (k / d) * (1 - 2 * j / J) for k in range(d)] for j in range(J)]
			model_input = model_input * l

		if FLAGS.interpolate == True:
			interpolate = tf.cast(tf.range(300), tf.float32) / 300.0
			interpolate = tf.expand_dims(interpolate, 0)
			interpolate = tf.expand_dims(interpolate, 2)  # (1, 300, 1)
			model_input = model_input * interpolate
		lstm_size = FLAGS.lstm_cells
		number_of_layers = FLAGS.lstm_layers

		stacked_lstm = tf.contrib.rnn.MultiRNNCell(
			[
				tf.contrib.rnn.BasicLSTMCell(
					lstm_size, forget_bias=1.0, state_is_tuple=False)
				for _ in range(number_of_layers)
			], state_is_tuple=False)

		loss = 0.0

		outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
		                                   sequence_length=num_frames,
		                                   dtype=tf.float32)

		# state = tf.nn.l2_normalize(state, dim=1)
		average_state = tf.nn.l2_normalize(tf.reduce_sum(model_input, axis=1), dim=1)
		state = tf.concat([state, average_state], 1)

		aggregated_model = getattr(video_level_models,
		                           FLAGS.video_level_classifier_model)

		return aggregated_model().create_model(
			model_input=state,
			vocab_size=vocab_size,
			**unused_params)


class SoftClusteringModel(models.BaseModel):
	def create_model(self, model_input, vocab_size, num_frames, **unused_params):
		input_features = slim.fully_connected(
			model_input,
			FLAGS.feature_dim,
			activation_fn=tf.nn.relu,
			weights_regularizer=slim.l2_regularizer(1e-8)
		)

		att_matrix = tf.matmul(input_features,
		                       tf.transpose(input_features, [0, 2, 1]))  # [batch_size, max_frames, max_frames]

		att_matrix = tf.expand_dims(att_matrix, -1)
		att = tf.reduce_sum(att_matrix, axis=2)  # [batch_size, max_frames]
		att = tf.nn.softmax(FLAGS.alpha * att)

		state = tf.reduce_sum(model_input * att, axis=1)  # [batch_size, num_features]
		state = tf.contrib.layers.layer_norm(
			inputs=state,
			center=True,
			scale=True,
			activation_fn=tf.nn.relu,
			trainable=True)

		# state = tf.nn.l2_normalize(state, dim=1)

		aggregated_model = getattr(video_level_models,
		                           FLAGS.video_level_classifier_model)

		return aggregated_model().create_model(
			model_input=state,
			vocab_size=vocab_size,
			**unused_params)


class RnnFvModel(models.BaseModel):
	def create_model(self, model_input, vocab_size, num_frames, **unused_params):
		"""Creates a model which uses a LSTM to predict the next element of the sequence.
		using derived gradient from the RNN as a vector representation,
		instead of using a hidden or an output layer of the RNN
		model_input: A 'batch_size' x 'sequence_length' x 'feature_size' matrix 
		"""
		lstm_size = FLAGS.lstm_cells

		feature_size = model_input.get_shape().as_list()[2]
		sequence_length = model_input.get_shape().as_list()[1]

		# start_token is important!
		start_token = tf.zeros_like(tf.expand_dims(model_input[:, 0, :], axis=1), dtype=tf.float32)
		input_sequence = tf.concat([start_token, model_input[:, :-1, :]], axis=1)
		output_sequence = model_input[:, :, :]

		# fc-relu
		# input_sequence = tf.reshape(input_sequence, [-1, feature_size])
		# fc1 = tf.contrib.layers.fully_connected(input_sequence, lstm_size, activation_fn=tf.nn.relu)
		# input_sequence = tf.reshape(fc1, [-1, sequence_length, lstm_size])

		cell = tf.contrib.rnn.BasicLSTMCell(lstm_size)
		outputs, state = tf.nn.dynamic_rnn(
			cell=cell,
			inputs=input_sequence,
			sequence_length=None,
			parallel_iterations=128,
			dtype=tf.float32)  # output = (batch, num_frames, lstm_size)

		# fc-linear
		outputs = tf.reshape(outputs, [-1, lstm_size])
		fc2 = tf.contrib.layers.fully_connected(outputs, feature_size, activation_fn=None)
		outputs = tf.reshape(fc2, [-1, sequence_length, feature_size])

		loss = tf.nn.l2_loss(outputs - output_sequence)

		dummy_pooled = tf.reduce_sum(model_input, axis=[1])
		dummy_output = slim.fully_connected(
			dummy_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
			weights_regularizer=slim.l2_regularizer(1e-8))

		return {"predictions": dummy_output, "loss": loss}


class LightVLAD():
	def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
		self.feature_size = feature_size
		self.max_frames = max_frames
		self.is_training = is_training
		self.add_batch_norm = add_batch_norm
		self.cluster_size = cluster_size

	def forward(self, reshaped_input):

		cluster_weights = tf.get_variable("cluster_weights",
		                                  [self.feature_size, self.cluster_size],
		                                  initializer=tf.random_normal_initializer(
			                                  stddev=1 / math.sqrt(self.feature_size)))

		activation = tf.matmul(reshaped_input, cluster_weights)

		if self.add_batch_norm:
			activation = slim.batch_norm(
				activation,
				center=True,
				scale=True,
				is_training=self.is_training,
				scope="cluster_bn")
		else:
			cluster_biases = tf.get_variable("cluster_biases",
			                                 [self.cluster_size],
			                                 initializer=tf.random_normal_initializer(
				                                 stddev=1 / math.sqrt(self.feature_size)))
			tf.summary.histogram("cluster_biases", cluster_biases)
			activation += cluster_biases

		activation = tf.nn.softmax(activation)

		activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

		activation = tf.transpose(activation, perm=[0, 2, 1])

		reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
		vlad = tf.matmul(activation, reshaped_input)

		vlad = tf.transpose(vlad, perm=[0, 2, 1])
		vlad = tf.nn.l2_normalize(vlad, 1)

		vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
		vlad = tf.nn.l2_normalize(vlad, 1)

		return vlad


class NetVLAD():
	def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
		self.feature_size = feature_size
		self.max_frames = max_frames
		self.is_training = is_training
		self.add_batch_norm = add_batch_norm
		self.cluster_size = cluster_size

	def forward(self, reshaped_input):

		cluster_weights = tf.get_variable("cluster_weights",
		                                  [self.feature_size, self.cluster_size],
		                                  initializer=tf.random_normal_initializer(
			                                  stddev=1 / math.sqrt(self.feature_size)))

		tf.summary.histogram("cluster_weights", cluster_weights)
		activation = tf.matmul(reshaped_input, cluster_weights)

		if self.add_batch_norm:
			activation = slim.batch_norm(
				activation,
				center=True,
				scale=True,
				is_training=self.is_training,
				scope="cluster_bn")
		else:
			cluster_biases = tf.get_variable("cluster_biases",
			                                 [self.cluster_size],
			                                 initializer=tf.random_normal_initializer(
				                                 stddev=1 / math.sqrt(self.feature_size)))
			tf.summary.histogram("cluster_biases", cluster_biases)
			activation += cluster_biases

		activation = tf.nn.softmax(activation)
		tf.summary.histogram("cluster_output", activation)

		activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

		a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

		cluster_weights2 = tf.get_variable("cluster_weights2",
		                                   [1, self.feature_size, self.cluster_size],
		                                   initializer=tf.random_normal_initializer(
			                                   stddev=1 / math.sqrt(self.feature_size)))

		a = tf.multiply(a_sum, cluster_weights2)

		activation = tf.transpose(activation, perm=[0, 2, 1])

		reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
		vlad = tf.matmul(activation, reshaped_input)
		vlad = tf.transpose(vlad, perm=[0, 2, 1])
		vlad = tf.subtract(vlad, a)

		vlad = tf.nn.l2_normalize(vlad, 1)

		vlad = tf.reshape(vlad, [-1, self.cluster_size * self.feature_size])
		vlad = tf.nn.l2_normalize(vlad, 1)

		return vlad


class NetVLAGD():
	def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
		self.feature_size = feature_size
		self.max_frames = max_frames
		self.is_training = is_training
		self.add_batch_norm = add_batch_norm
		self.cluster_size = cluster_size

	def forward(self, reshaped_input):

		cluster_weights = tf.get_variable("cluster_weights",
		                                  [self.feature_size, self.cluster_size],
		                                  initializer=tf.random_normal_initializer(
			                                  stddev=1 / math.sqrt(self.feature_size)))

		activation = tf.matmul(reshaped_input, cluster_weights)

		if self.add_batch_norm:
			activation = slim.batch_norm(
				activation,
				center=True,
				scale=True,
				is_training=self.is_training,
				scope="cluster_bn")
		else:
			cluster_biases = tf.get_variable("cluster_biases",
			                                 [self.cluster_size],
			                                 initializer=tf.random_normal_initializer(
				                                 stddev=1 / math.sqrt(self.feature_size)))

		activation = tf.nn.softmax(activation)

		activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

		gate_weights = tf.get_variable("gate_weights",
		                               [1, self.cluster_size, self.feature_size],
		                               initializer=tf.random_normal_initializer(
			                               stddev=1 / math.sqrt(self.feature_size)))

		gate_weights = tf.sigmoid(gate_weights)

		activation = tf.transpose(activation, perm=[0, 2, 1])

		reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])

		vlagd = tf.matmul(activation, reshaped_input)
		vlagd = tf.multiply(vlagd, gate_weights)

		vlagd = tf.transpose(vlagd, perm=[0, 2, 1])

		vlagd = tf.nn.l2_normalize(vlagd, 1)

		vlagd = tf.reshape(vlagd, [-1, self.cluster_size * self.feature_size])
		vlagd = tf.nn.l2_normalize(vlagd, 1)

		return vlagd


class NetVLADModelLF(models.BaseModel):
	"""Creates a NetVLAD based model.
	Args:
	  model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
				   input features.
	  vocab_size: The number of classes in the dataset.
	  num_frames: A vector of length 'batch' which indicates the number of
		   frames for each video (before padding).
	Returns:
	  A dictionary with a tensor containing the probability predictions of the
	  model in the 'predictions' key. The dimensions of the tensor are
	  'batch_size' x 'num_classes'.
	"""

	def create_model(self,
	                 model_input,
	                 vocab_size,
	                 num_frames,
	                 iterations=None,
	                 add_batch_norm=None,
	                 sample_random_frames=None,
	                 cluster_size=None,
	                 hidden_size=None,
	                 is_training=True,
	                 **unused_params):
		iterations = iterations or FLAGS.iterations
		add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
		random_frames = sample_random_frames or FLAGS.sample_random_frames
		cluster_size = cluster_size or FLAGS.netvlad_cluster_size
		hidden1_size = hidden_size or FLAGS.netvlad_hidden_size
		relu = FLAGS.netvlad_relu
		dimred = FLAGS.netvlad_dimred
		gating = FLAGS.gating
		remove_diag = FLAGS.gating_remove_diag
		lightvlad = FLAGS.lightvlad
		vlagd = FLAGS.vlagd


		num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)

		#num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float16)
		if random_frames:
			model_input = utils.SampleRandomFrames(model_input, num_frames,
			                                       iterations)
		else:
			model_input = utils.SampleRandomSequence(model_input, num_frames,
			                                         iterations)



		# model_input = slim.fully_connected(
		# 	model_input, 1024, activation_fn=tf.nn.sigmoid,
		# 	weights_regularizer=slim.l2_regularizer(1e-8))

		max_frames = model_input.get_shape().as_list()[1]


		feature_size = model_input.get_shape().as_list()[2]


		reshaped_input = tf.reshape(model_input, [-1, feature_size])

		if lightvlad:
			video_NetVLAD = LightVLAD(1024, max_frames, cluster_size, add_batch_norm, is_training)
			#audio_NetVLAD = LightVLAD(128, max_frames, cluster_size / 2, add_batch_norm, is_training)
		elif vlagd:
			video_NetVLAD = NetVLAGD(1024, max_frames, cluster_size, add_batch_norm, is_training)
			#audio_NetVLAD = NetVLAGD(128, max_frames, cluster_size / 2, add_batch_norm, is_training)
		else:
			video_NetVLAD = NetVLAD(1024, max_frames, cluster_size, add_batch_norm, is_training)
			#audio_NetVLAD = NetVLAD(128, max_frames, cluster_size / 2, add_batch_norm, is_training)

		if add_batch_norm:  # and not lightvlad:
			reshaped_input = slim.batch_norm(
				reshaped_input,
				center=True,
				scale=True,
				is_training=is_training,
				scope="input_bn")

		with tf.variable_scope("video_VLAD"):
			vlad_video = video_NetVLAD.forward(reshaped_input[:, 0:1024])

		# with tf.variable_scope("audio_VLAD"):
		# 	vlad_audio = audio_NetVLAD.forward(reshaped_input[:, 1024:])

		#vlad = tf.concat([vlad_video, vlad_audio], 1)
		vlad = tf.concat([vlad_video], 1)

		vlad_dim = vlad.get_shape().as_list()[1]
		hidden1_weights = tf.get_variable("hidden1_weights",
		                                  [vlad_dim, hidden1_size],
		                                  initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))

		activation = tf.matmul(vlad, hidden1_weights)

		if add_batch_norm and relu:
			activation = slim.batch_norm(
				activation,
				center=True,
				scale=True,
				is_training=is_training,
				scope="hidden1_bn")

		else:
			hidden1_biases = tf.get_variable("hidden1_biases",
			                                 [hidden1_size],
			                                 initializer=tf.random_normal_initializer(stddev=0.01))
			tf.summary.histogram("hidden1_biases", hidden1_biases)
			activation += hidden1_biases

		if relu:
			activation = tf.nn.relu6(activation)

		if gating:
			gating_weights = tf.get_variable("gating_weights_2",
			                                 [hidden1_size, hidden1_size],
			                                 initializer=tf.random_normal_initializer(
				                                 stddev=1 / math.sqrt(hidden1_size)))

			gates = tf.matmul(activation, gating_weights)

			if remove_diag:
				# removes diagonals coefficients
				diagonals = tf.matrix_diag_part(gating_weights)
				gates = gates - tf.multiply(diagonals, activation)

			if add_batch_norm:
				gates = slim.batch_norm(
					gates,
					center=True,
					scale=True,
					is_training=is_training,
					scope="gating_bn")
			else:
				gating_biases = tf.get_variable("gating_biases",
				                                [cluster_size],
				                                initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
				gates += gating_biases

			gates = tf.sigmoid(gates)

			activation = tf.multiply(activation, gates)

		aggregated_model = getattr(video_level_models,
		                           FLAGS.video_level_classifier_model)

		return aggregated_model().create_model(
			model_input=activation,
			vocab_size=vocab_size,
			is_training=is_training,
			**unused_params)


class NetFV():
	def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
		self.feature_size = feature_size
		self.max_frames = max_frames
		self.is_training = is_training
		self.add_batch_norm = add_batch_norm
		self.cluster_size = cluster_size

	def forward(self, reshaped_input):
		cluster_weights = tf.get_variable("cluster_weights",
		                                  [self.feature_size, self.cluster_size],
		                                  initializer=tf.random_normal_initializer(
			                                  stddev=1 / math.sqrt(self.feature_size)))

		covar_weights = tf.get_variable("covar_weights",
		                                [self.feature_size, self.cluster_size],
		                                initializer=tf.random_normal_initializer(mean=1.0, stddev=1 / math.sqrt(
			                                self.feature_size)))

		covar_weights = tf.square(covar_weights)
		eps = tf.constant([1e-6])
		covar_weights = tf.add(covar_weights, eps)

		tf.summary.histogram("cluster_weights", cluster_weights)
		activation = tf.matmul(reshaped_input, cluster_weights)
		if self.add_batch_norm:
			activation = slim.batch_norm(
				activation,
				center=True,
				scale=True,
				is_training=self.is_training,
				scope="cluster_bn")
		else:
			cluster_biases = tf.get_variable("cluster_biases",
			                                 [self.cluster_size],
			                                 initializer=tf.random_normal(stddev=1 / math.sqrt(self.feature_size)))
			tf.summary.histogram("cluster_biases", cluster_biases)
			activation += cluster_biases

		activation = tf.nn.softmax(activation)
		tf.summary.histogram("cluster_output", activation)

		activation = tf.reshape(activation, [-1, self.max_frames, self.cluster_size])

		a_sum = tf.reduce_sum(activation, -2, keep_dims=True)

		if not FLAGS.fv_couple_weights:
			cluster_weights2 = tf.get_variable("cluster_weights2",
			                                   [1, self.feature_size, self.cluster_size],
			                                   initializer=tf.random_normal_initializer(
				                                   stddev=1 / math.sqrt(self.feature_size)))
		else:
			cluster_weights2 = tf.scalar_mul(FLAGS.fv_coupling_factor, cluster_weights)

		a = tf.multiply(a_sum, cluster_weights2)

		activation = tf.transpose(activation, perm=[0, 2, 1])

		reshaped_input = tf.reshape(reshaped_input, [-1, self.max_frames, self.feature_size])
		fv1 = tf.matmul(activation, reshaped_input)

		fv1 = tf.transpose(fv1, perm=[0, 2, 1])

		# computing second order FV
		a2 = tf.multiply(a_sum, tf.square(cluster_weights2))

		b2 = tf.multiply(fv1, cluster_weights2)
		fv2 = tf.matmul(activation, tf.square(reshaped_input))

		fv2 = tf.transpose(fv2, perm=[0, 2, 1])
		fv2 = tf.add_n([a2, fv2, tf.scalar_mul(-2, b2)])

		fv2 = tf.divide(fv2, tf.square(covar_weights))
		fv2 = tf.subtract(fv2, a_sum)

		fv2 = tf.reshape(fv2, [-1, self.cluster_size * self.feature_size])

		fv2 = tf.nn.l2_normalize(fv2, 1)
		fv2 = tf.reshape(fv2, [-1, self.cluster_size * self.feature_size])
		fv2 = tf.nn.l2_normalize(fv2, 1)

		fv1 = tf.subtract(fv1, a)
		fv1 = tf.divide(fv1, covar_weights)

		fv1 = tf.nn.l2_normalize(fv1, 1)
		fv1 = tf.reshape(fv1, [-1, self.cluster_size * self.feature_size])
		fv1 = tf.nn.l2_normalize(fv1, 1)

		return tf.concat([fv1, fv2], 1)


class NetFVModelLF(models.BaseModel):
	"""Creates a NetFV based model.
	   It emulates a Gaussian Mixture Fisher Vector pooling operations
  
	Args:
	  model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
				   input features.
	  vocab_size: The number of classes in the dataset.
	  num_frames: A vector of length 'batch' which indicates the number of
		   frames for each video (before padding).
  
	Returns:
	  A dictionary with a tensor containing the probability predictions of the
	  model in the 'predictions' key. The dimensions of the tensor are
	  'batch_size' x 'num_classes'.
	"""

	def create_model(self,
	                 model_input,
	                 vocab_size,
	                 num_frames,
	                 iterations=None,
	                 add_batch_norm=None,
	                 sample_random_frames=None,
	                 cluster_size=None,
	                 hidden_size=None,
	                 is_training=True,
	                 **unused_params):
		iterations = iterations or FLAGS.iterations
		add_batch_norm = add_batch_norm or FLAGS.netvlad_add_batch_norm
		random_frames = sample_random_frames or FLAGS.sample_random_frames
		cluster_size = cluster_size or FLAGS.fv_cluster_size
		hidden1_size = hidden_size or FLAGS.fv_hidden_size
		relu = FLAGS.fv_relu
		gating = FLAGS.gating

		num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float16)
		if random_frames:
			model_input = utils.SampleRandomFrames(model_input, num_frames,
			                                       iterations)
		else:
			model_input = utils.SampleRandomSequence(model_input, num_frames,
			                                         iterations)


		max_frames = model_input.get_shape().as_list()[1]
		feature_size = model_input.get_shape().as_list()[2]
		reshaped_input = tf.reshape(model_input, [-1, feature_size])
		tf.summary.histogram("input_hist", reshaped_input)

		video_NetFV = NetFV(1024, max_frames, cluster_size, add_batch_norm, is_training)
		#audio_NetFV = NetFV(128, max_frames, cluster_size / 2, add_batch_norm, is_training)

		if add_batch_norm:
			reshaped_input = slim.batch_norm(
				reshaped_input,
				center=True,
				scale=True,
				is_training=is_training,
				scope="input_bn")

		with tf.variable_scope("video_FV"):
			fv_video = video_NetFV.forward(reshaped_input[:, 0:8192])

		# with tf.variable_scope("audio_FV"):
		# 	fv_audio = audio_NetFV.forward(reshaped_input[:, 1024:])

		#fv = tf.concat([fv_video, fv_audio], 1)
		fv = tf.concat([fv_video], 1)

		fv_dim = fv.get_shape().as_list()[1]
		hidden1_weights = tf.get_variable("hidden1_weights",
		                                  [fv_dim, hidden1_size],
		                                  initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))

		activation = tf.matmul(fv, hidden1_weights)

		if add_batch_norm and relu:
			activation = slim.batch_norm(
				activation,
				center=True,
				scale=True,
				is_training=is_training,
				scope="hidden1_bn")
		else:
			hidden1_biases = tf.get_variable("hidden1_biases",
			                                 [hidden1_size],
			                                 initializer=tf.random_normal_initializer(stddev=0.01))
			tf.summary.histogram("hidden1_biases", hidden1_biases)
			activation += hidden1_biases

		if relu:
			activation = tf.nn.relu6(activation)

		if gating:
			gating_weights = tf.get_variable("gating_weights_2",
			                                 [hidden1_size, hidden1_size],
			                                 initializer=tf.random_normal_initializer(
				                                 stddev=1 / math.sqrt(hidden1_size)))

			gates = tf.matmul(activation, gating_weights)

			if add_batch_norm:
				gates = slim.batch_norm(
					gates,
					center=True,
					scale=True,
					is_training=is_training,
					scope="gating_bn")
			else:
				gating_biases = tf.get_variable("gating_biases",
				                                [cluster_size],
				                                initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
				gates += gating_biases

			gates = tf.sigmoid(gates)

			activation = tf.multiply(activation, gates)

		aggregated_model = getattr(video_level_models,
		                           FLAGS.video_level_classifier_model)

		return aggregated_model().create_model(
			model_input=activation,
			vocab_size=vocab_size,
			is_training=is_training,
			**unused_params)