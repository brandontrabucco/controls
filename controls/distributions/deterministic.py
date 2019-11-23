"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.distributions.distribution import Distribution
import tensorflow as tf


class Deterministic(Distribution):

    def sample(
            self,
            time,
            inputs
    ):
        """Sample from a deterministic variable.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the deterministic distribution
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - samples: samples from a deterministic distribution
            with shape [batch_dim, outputs_dim, 1]
        - log_prob: the log probability of samples
            with shape [batch_dim, 1, 1].
        """
        samples = self.get_parameters(time, inputs)
        return samples, tf.zeros([tf.shape(samples)[0], 1, 1])

    def expected_value(
            self,
            time,
            inputs
    ):
        """Expected value of a deterministic variable.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the deterministic distribution
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - samples: samples from a deterministic distribution
            with shape [batch_dim, outputs_dim, 1]
        - log_prob: the log probability of samples
            with shape [batch_dim, 1, 1].
        """
        samples = self.get_parameters(time, inputs)
        return samples, tf.zeros([tf.shape(samples)[0], 1, 1])

    def log_prob(
            self,
            samples,
            time,
            inputs
    ):
        """Log probability under a deterministic variable.

        Args:
        - samples: samples from a deterministic distribution
            with shape [batch_dim, outputs_dim, 1]
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the deterministic distribution
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - log_prob: the log probability of samples
            with shape [batch_dim, 1, 1].
        """
        return tf.zeros([tf.shape(samples)[0], 1, 1])
