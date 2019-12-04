"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.distributions.gaussian import Gaussian
import tensorflow as tf


class TanhGaussian(Gaussian):

    def fit(
            self,
            tanh_samples
    ):
        """Maximum likelihood estimation of the distribution.

        Args:
        - samples: samples from a distribution of a random variable
            with shape [T, batch_dim, outputs_dim].

        Returns:
        - distribution: a distribution fitted using maximum likelihoods estimation
            the function returns tensors with shape [batch_dim, output_dim].
        """
        samples = tf.math.atanh(tf.clip_by_value(tanh_samples, -0.999, 0.999))
        mean = tf.reduce_mean(samples, axis=2)
        error = samples - mean[:, :, tf.newaxis, :]
        covariance = tf.matmul(error, error, transpose_a=True) / tf.cast(tf.shape(samples)[2], tf.float32)
        covariance = covariance + 1e-3 * tf.eye(tf.shape(mean)[1], batch_shape=tf.shape(samples)[:2])

        std = tf.linalg.sqrtm(covariance)
        precision = tf.linalg.inv(covariance)
        log_determinant = tf.linalg.logdet(covariance)

        return TanhGaussian(lambda time, inputs: (
            mean[time], std[time], precision[time], log_determinant[time]))

    def sample(
            self,
            time,
            inputs
    ):
        """Sample from a tanh gaussian.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the tanh gaussian distribution
            with shape [batch_dim, inputs_dim[i]].

        Returns:
        - samples: samples from a tanh gaussian distribution
            with shape [batch_dim, outputs_dim]
        - log_prob: the log probability of samples
            with shape [batch_dim].
        """
        gaussian_samples, log_probs = Gaussian.sample(time, inputs)
        tanh_samples = tf.tanh(gaussian_samples)
        return tanh_samples, log_probs - tf.reduce_sum(
            2.0 * (tf.math.log(2.0) - gaussian_samples - tf.math.softplus(
                -2.0 * gaussian_samples)), axis=(-1))

    def expected_value(
            self,
            time,
            inputs
    ):
        """Expected value of a tanh gaussian.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the tanh gaussian distribution
            with shape [batch_dim, inputs_dim[i]].

        Returns:
        - samples: samples from a tanh gaussian distribution
            with shape [batch_dim, outputs_dim]
        - log_prob: the log probability of samples
            with shape [batch_dim].
        """
        gaussian_samples, log_probs = Gaussian.expected_value(time, inputs)
        tanh_samples = tf.tanh(gaussian_samples)
        return tanh_samples, log_probs - tf.reduce_sum(
            2.0 * (tf.math.log(2.0) - gaussian_samples - tf.math.softplus(
                -2.0 * gaussian_samples)), axis=(-1))

    def log_prob(
            self,
            tanh_samples,
            time,
            inputs
    ):
        """Log probability under a tanh gaussian.

        Args:
        - tanh_samples: samples from a tanh gaussian distribution
            with shape [batch_dim, outputs_dim]
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the tanh gaussian distribution
            with shape [batch_dim, inputs_dim[i]].

        Returns:
        - log_prob: the log probability of samples
            with shape [batch_dim].
        """
        gaussian_samples = tf.math.atanh(tf.clip_by_value(tanh_samples, -0.999, 0.999))
        log_probs = Gaussian.log_prob(gaussian_samples, time, inputs)
        return tanh_samples, log_probs - tf.reduce_sum(
            2.0 * (tf.math.log(2.0) - gaussian_samples - tf.math.softplus(
                -2.0 * gaussian_samples)), axis=(-1))
