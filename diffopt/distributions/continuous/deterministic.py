"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from diffopt.distributions.distribution import Distribution
import tensorflow as tf


class Deterministic(Distribution):

    def fit(
            self,
            samples
    ):
        """Maximum likelihood estimation of the distribution.

        Args:
        - samples: samples from a distribution of a random variable
            with shape [T, batch_dim, outputs_dim].

        Returns:
        - distribution: a distribution fitted using maximum likelihoods estimation
            the function returns tensors with shape [batch_dim, output_dim].
        """
        return NotImplemented

    def sample(
            self,
            time,
            inputs
    ):
        """Sample from a deterministic variable.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the deterministic distribution
            with shape [batch_dim, inputs_dim[i]].

        Returns:
        - samples: samples from a deterministic distribution
            with shape [batch_dim, outputs_dim]
        - log_prob: the log probability of samples
            with shape [batch_dim].
        """
        samples = self.get_parameters(time, inputs)[0]
        return samples, tf.zeros([tf.shape(samples)[0]])

    def expected_value(
            self,
            time,
            inputs
    ):
        """Expected value of a deterministic variable.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the deterministic distribution
            with shape [batch_dim, inputs_dim[i]].

        Returns:
        - samples: samples from a deterministic distribution
            with shape [batch_dim, outputs_dim]
        - log_prob: the log probability of samples
            with shape [batch_dim].
        """
        samples = self.get_parameters(time, inputs)[0]
        return samples, tf.zeros([tf.shape(samples)[0]])

    def log_prob(
            self,
            samples,
            time,
            inputs
    ):
        """Log probability under a deterministic variable.

        Args:
        - samples: samples from a deterministic distribution
            with shape [batch_dim, outputs_dim]
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the deterministic distribution
            with shape [batch_dim, inputs_dim[i]].

        Returns:
        - log_prob: the log probability of samples
            with shape [batch_dim].
        """
        return tf.zeros([tf.shape(samples)[0]])
