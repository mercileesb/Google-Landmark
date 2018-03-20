# Copyright 2017 Antoine Miech All Rights Reserved.
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

"""Provides definitions for non-regularized training or test losses."""

import tensorflow as tf
from tensorflow import flags
import scipy.io as sio
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_float(
      "alpha", "0.95",
      "Ponderation for XENT")

flags.DEFINE_float("delta", 0.5, "delta value for Huber Loss")
flags.DEFINE_float("alfa", 0.95, "alfa parameter for center loss.")
flags.DEFINE_float("epsilon", 10e-6, "epsilon parameter for center loss.")
flags.DEFINE_float("beta", 1e-3, "beta parameter for center loss.")


class BaseLoss(object):
  """Inherit from this class when implementing new losses."""

  def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
    """Calculates the average loss of the examples in a mini-batch.

     Args:
      unused_predictions: a 2-d tensor storing the prediction scores, in which
        each row represents a sample in the mini-batch and each column
        represents a class.
      unused_labels: a 2-d tensor storing the labels, which has the same shape
        as the unused_predictions. The labels must be in the range of 0 and 1.
      unused_params: loss specific parameters.

    Returns:
      A scalar loss tensor.
    """
    raise NotImplementedError()


class CrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      alpha = FLAGS.alpha

      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = 2*(alpha*float_labels * tf.log(predictions + epsilon) + (1-alpha)*(
          1 - float_labels) * tf.log(1 - predictions + epsilon))
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


class L2_CrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
                                                                            1 - float_labels) * tf.log(
        1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      cross_entropy_loss = cross_entropy_loss * cross_entropy_loss
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


class Huber_CrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      delta = FLAGS.delta
      epsilon = 10e-6
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
                                                                            1 - float_labels) * tf.log(
        1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)

      # Huber Loss Approximation
      cross_entropy_loss = delta * delta * (
      tf.sqrt(1 + (cross_entropy_loss / delta) * (cross_entropy_loss / delta)) - 1)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


class HingeLoss(BaseLoss):
  """Calculate the hinge loss between the predictions and labels.
  Note the subgradient is used in the backpropagation, and thus the optimization
  may converge slower. The predictions trained by the hinge loss are between -1
  and +1.
  """

  def calculate_loss(self, predictions, labels, b=1.0, **unused_params):
    with tf.name_scope("loss_hinge"):
      float_labels = tf.cast(labels, tf.float32)
      all_zeros = tf.zeros(tf.shape(float_labels), dtype=tf.float32)
      all_ones = tf.ones(tf.shape(float_labels), dtype=tf.float32)
      sign_labels = tf.subtract(tf.scalar_mul(2, float_labels), all_ones)
      hinge_loss = tf.maximum(
        all_zeros, tf.scalar_mul(b, all_ones) - sign_labels * predictions)
      return tf.reduce_mean(tf.reduce_sum(hinge_loss, 1))


class L2_HingeLoss(BaseLoss):
  """Calculate the hinge loss between the predictions and labels.
  Note the subgradient is used in the backpropagation, and thus the optimization
  may converge slower. The predictions trained by the hinge loss are between -1
  and +1.
  """

  def calculate_loss(self, predictions, labels, b=1.0, **unused_params):
    with tf.name_scope("loss_hinge"):
      float_labels = tf.cast(labels, tf.float32)
      all_zeros = tf.zeros(tf.shape(float_labels), dtype=tf.float32)
      all_ones = tf.ones(tf.shape(float_labels), dtype=tf.float32)
      sign_labels = tf.subtract(tf.scalar_mul(2, float_labels), all_ones)
      hinge_loss = tf.maximum(
        all_zeros, tf.scalar_mul(b, all_ones) - sign_labels * predictions)
      hinge_loss = hinge_loss * hinge_loss
      return tf.reduce_mean(tf.reduce_sum(hinge_loss, 1))


class Huber_HingeLoss(BaseLoss):
  """Calculate the hinge loss between the predictions and labels.
  Note the subgradient is used in the backpropagation, and thus the optimization
  may converge slower. The predictions trained by the hinge loss are between -1
  and +1.
  """

  def calculate_loss(self, predictions, labels, b=1.0, **unused_params):
    with tf.name_scope("loss_hinge"):
      delta = FLAGS.delta
      float_labels = tf.cast(labels, tf.float32)
      all_zeros = tf.zeros(tf.shape(float_labels), dtype=tf.float32)
      all_ones = tf.ones(tf.shape(float_labels), dtype=tf.float32)
      sign_labels = tf.subtract(tf.scalar_mul(2, float_labels), all_ones)
      hinge_loss = tf.maximum(
        all_zeros, tf.scalar_mul(b, all_ones) - sign_labels * predictions)

      # Huber Loss Approximation
      hinge_loss_entropy_loss = delta * delta * (tf.sqrt(1 + (hinge_loss / delta) * (hinge_loss / delta)) - 1)

      return tf.reduce_mean(tf.reduce_sum(hinge_loss, 1))


class SoftmaxLoss(BaseLoss):
  """Calculate the softmax loss between the predictions and labels.
  The function calculates the loss in the following way: first we feed the
  predictions to the softmax activation function and then we calculate
  the minus linear dot product between the logged softmax activations and the
  normalized ground truth label.
  It is an extension to the one-hot label. It allows for more than one positive
  labels for each sample.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_softmax"):
      epsilon = 10e-8
      float_labels = tf.cast(labels, tf.float32)
      # l1 normalization (labels are no less than 0)
      label_rowsum = tf.maximum(
        tf.reduce_sum(float_labels, 1, keep_dims=True),
        epsilon)
      norm_float_labels = tf.div(float_labels, label_rowsum)
      softmax_outputs = tf.nn.softmax(predictions)
      softmax_loss = tf.negative(tf.reduce_sum(
        tf.multiply(norm_float_labels, tf.log(softmax_outputs)), 1))
    return tf.reduce_mean(softmax_loss)


class CenterLoss(BaseLoss):
  """Calculate the Centet loss using bottleneck feature vector.
  loss = cross_entropy_loss + 0.001 (beta) * center_loss
  The function calculates the loss in the following way: 
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_center"):
      epsilon = FLAGS.epsilon
      alfa = FLAGS.alfa
      beta = FLAGS.beta

      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
                                                                            1 - float_labels) * tf.log(
        1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      xent_loss = tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))

      features = unused_params['video_feature']
      centers = tf.get_variable('centers', [labels.shape[1], features.shape[1]], dtype=tf.float32,
                                initializer=tf.constant_initializer(0), trainable=False)

      multi_label = tf.where(tf.equal(float_labels, 1))

      feature_index = tf.squeeze(tf.slice(multi_label, [0, 0], [-1, 1]))
      label_index = tf.squeeze(tf.slice(multi_label, [0, 1], [-1, 1]))

      features_batch = tf.gather(features, feature_index)
      centers_batch = tf.gather(centers, label_index)
      diff = (1 - alfa) * (centers_batch - features_batch)
      centers = tf.scatter_sub(centers, label_index, diff)
      # center_loss = tf.nn.l2_loss(features_batch - centers_batch)
      center_loss = tf.reduce_mean(tf.squared_difference(features_batch, centers_batch))

      loss = xent_loss + beta * center_loss

      return loss