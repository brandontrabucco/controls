"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.distributions.distribution import Distribution
import tensorflow as tf
import math


class Gaussian(Distribution):

    def sample(
            self,
            time,
            inputs
    ):
        """Sample from a gaussian.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the linear gaussian distribution
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - samples: samples from a gaussian distribution
            with shape [batch_dim, outputs_dim, 1]
        - log_prob: the log probability of samples
            with shape [batch_dim, 1, 1].
        """
        mean, std, precision, log_determinant = self.get_parameters(time, inputs)
        samples = mean + std @ tf.random.normal(tf.shape(mean), dtype=mean.dtype)
        error = samples - mean
        return samples, -0.5 * (
            tf.matmul(tf.matmul(error, precision, transpose_a=True), error) +
            log_determinant +
            tf.math.log(2 * math.pi) * tf.cast(tf.shape(samples)[1], mean.dtype))

    def expected_value(
            self,
            time,
            inputs
    ):
        """Expected value of a gaussian.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the linear gaussian distribution
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - samples: samples from a gaussian distribution
            with shape [batch_dim, outputs_dim, 1]
        - log_prob: the log probability of samples
            with shape [batch_dim, 1, 1].
        """
        mean, std, precision, log_determinant = self.get_parameters(time, inputs)
        return mean, -0.5 * (
            log_determinant +
            tf.math.log(2 * math.pi) * tf.cast(tf.shape(mean)[1], mean.dtype))

    def log_prob(
            self,
            samples,
            time,
            inputs
    ):
        """Log probability under a gaussian.

        Args:
        - samples: samples from a gaussian distribution
            with shape [batch_dim, outputs_dim, 1]
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the linear gaussian distribution
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - log_prob: the log probability of samples
            with shape [batch_dim, 1, 1].
        """
        mean, std, precision, log_determinant = self.get_parameters(time, inputs)
        error = samples - mean
        return -0.5 * (
            tf.matmul(tf.matmul(error, precision, transpose_a=True), error) +
            log_determinant +
            tf.math.log(2 * math.pi) * tf.cast(tf.shape(mean)[1], mean.dtype))
