"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.distributions.gaussian import Gaussian
import tensorflow as tf


class TanhGaussian(Gaussian):

    def sample(
            self,
            time,
            inputs
    ):
        """Sample from a tanh gaussian.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the tanh gaussian distribution
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - samples: samples from a tanh gaussian distribution
            with shape [batch_dim, outputs_dim, 1]
        - log_prob: the log probability of samples
            with shape [batch_dim, 1, 1].
        """
        gaussian_samples, log_probs = Gaussian.sample(time, inputs)
        tanh_samples = tf.tanh(gaussian_samples)
        return tanh_samples, log_probs - tf.reduce_sum(
            2.0 * (tf.math.log(2.0) - gaussian_samples - tf.math.softplus(
                -2.0 * gaussian_samples)), axis=(-2), keepdims=True)

    def expected_value(
            self,
            time,
            inputs
    ):
        """Expected value of a tanh gaussian.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the tanh gaussian distribution
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - samples: samples from a tanh gaussian distribution
            with shape [batch_dim, outputs_dim, 1]
        - log_prob: the log probability of samples
            with shape [batch_dim, 1, 1].
        """
        gaussian_samples, log_probs = Gaussian.expected_value(time, inputs)
        tanh_samples = tf.tanh(gaussian_samples)
        return tanh_samples, log_probs - tf.reduce_sum(
            2.0 * (tf.math.log(2.0) - gaussian_samples - tf.math.softplus(
                -2.0 * gaussian_samples)), axis=(-2), keepdims=True)

    def log_prob(
            self,
            tanh_samples,
            time,
            inputs
    ):
        """Log probability under a tanh gaussian.

        Args:
        - tanh_samples: samples from a tanh gaussian distribution
            with shape [batch_dim, outputs_dim, 1]
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the tanh gaussian distribution
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - log_prob: the log probability of samples
            with shape [batch_dim, 1, 1].
        """
        gaussian_samples = tf.math.atanh(tf.clip_by_value(tanh_samples, -0.999, 0.999))
        log_probs = Gaussian.log_prob(gaussian_samples, time, inputs)
        return tanh_samples, log_probs - tf.reduce_sum(
            2.0 * (tf.math.log(2.0) - gaussian_samples - tf.math.softplus(
                -2.0 * gaussian_samples)), axis=(-2), keepdims=True)
