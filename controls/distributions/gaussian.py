"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.distributions.distribution import Distribution
import tensorflow as tf
import math


class Gaussian(Distribution):

    def fit(
            self,
            samples
    ):
        """Maximum likelihood estimation of the distribution.

        Args:
        - samples: samples from a distribution of a random variable
            with shape [T, batch_dim, examples, outputs_dim].

        Returns:
        - distribution: a distribution fitted using maximum likelihoods estimation
            the function returns tensors with shape [batch_dim, output_dim].
        """
        mean = tf.reduce_mean(samples, axis=2)
        error = samples - mean[:, :, tf.newaxis, :]
        covariance = tf.matmul(error, error, transpose_a=True) / tf.cast(tf.shape(samples)[2], tf.float32)
        covariance = covariance + 1e-3 * tf.eye(tf.shape(mean)[1], batch_shape=tf.shape(samples)[:2])

        std = tf.linalg.sqrtm(covariance)
        precision = tf.linalg.inv(covariance)
        log_determinant = tf.linalg.logdet(covariance)

        return Gaussian(lambda time, inputs: (
            mean[time], std[time], precision[time], log_determinant[time]))

    def sample(
            self,
            time,
            inputs
    ):
        """Sample from a gaussian.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the gaussian distribution
            with shape [batch_dim, inputs_dim[i]].

        Returns:
        - samples: samples from a gaussian distribution
            with shape [batch_dim, outputs_dim]
        - log_prob: the log probability of samples
            with shape [batch_dim].
        """
        mean, std, precision, log_determinant = self.get_parameters(time, inputs)
        samples = mean[:, :, tf.newaxis] + std @ tf.random.normal(tf.shape(mean), dtype=mean.dtype)[:, :, tf.newaxis]
        error = samples - mean[:, :, tf.newaxis]
        return samples[:, :, 0], -0.5 * (
            tf.matmul(tf.matmul(error, precision, transpose_a=True), error)[:, 0, 0] +
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
        - inputs[i]: the inputs to the gaussian distribution
            with shape [batch_dim, inputs_dim[i]].

        Returns:
        - samples: samples from a gaussian distribution
            with shape [batch_dim, outputs_dim]
        - log_prob: the log probability of samples
            with shape [batch_dim].
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
            with shape [batch_dim, outputs_dim]
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the gaussian distribution
            with shape [batch_dim, inputs_dim[i]].

        Returns:
        - log_prob: the log probability of samples
            with shape [batch_dim].
        """
        mean, std, precision, log_determinant = self.get_parameters(time, inputs)
        error = samples[:, :, tf.newaxis] - mean[:, :, tf.newaxis]
        return -0.5 * (
            tf.matmul(tf.matmul(error, precision, transpose_a=True), error)[:, 0, 0] +
            log_determinant +
            tf.math.log(2 * math.pi) * tf.cast(tf.shape(mean)[1], mean.dtype))
