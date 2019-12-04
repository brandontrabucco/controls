"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.distributions.distribution import Distribution
import tensorflow as tf


class Categorical(Distribution):

    def fit(
            self,
            samples
    ):
        """Maximum likelihood estimation of the distribution.

        Args:
        - samples: samples from a distribution of a random variable
            with shape [T, batch_dim, examples, 1].

        Returns:
        - distribution: a distribution fitted using maximum likelihoods estimation
            the function returns tensors with shape [batch_dim, 1].
        """
        max_value = tf.cast(tf.reduce_max(samples), tf.int32) + 1
        horizon = tf.shape(samples)[0]
        batch_dim = tf.shape(samples)[1]

        def histogram_body(out, idx):
            return out.write(idx, tf.histogram_fixed_width(
                samples[idx // batch_dim, idx % batch_dim, :, 0],
                tf.cast([0, max_value], tf.int32),
                max_value, dtype=tf.int32)), idx + 1

        distribution = tf.reshape(tf.while_loop(
            lambda out, idx: tf.less(idx, horizon * batch_dim),
            histogram_body,
            (tf.TensorArray(tf.int32, size=tf.shape(samples)[0]),
                0))[0].stack(), [horizon, batch_dim, max_value])

        return Categorical(lambda time, inputs: [
            tf.math.log(1e-10 + tf.cast(distribution[time, :, :], tf.float32))])

    def sample(
            self,
            time,
            inputs
    ):
        """Sample from a categorical distribution.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the categorical distribution
            with shape [batch_dim, inputs_dim[i]].

        Returns:
        - samples: samples from a categorical distribution
            with shape [batch_dim, 1]
        - log_prob: the log probability of samples
            with shape [batch_dim].
        """
        log_probs = self.get_parameters(time, inputs)[0]
        categorical_samples = tf.random.categorical(log_probs, 1, dtype=tf.int32)
        return categorical_samples, tf.gather_nd(
            log_probs, categorical_samples, batch_dims=1)

    def expected_value(
            self,
            time,
            inputs
    ):
        """Expected value of a categorical distribution.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the categorical distribution
            with shape [batch_dim, inputs_dim[i]].

        Returns:
        - samples: samples from a categorical distribution
            with shape [batch_dim, 1]
        - log_prob: the log probability of samples
            with shape [batch_dim].
        """
        log_probs = self.get_parameters(time, inputs)[0]
        categorical_samples = tf.argmax(log_probs, axis=(-1), output_type=tf.int32)[:, tf.newaxis]
        return categorical_samples, tf.gather_nd(
            log_probs, categorical_samples, batch_dims=1)

    def log_prob(
            self,
            categorical_samples,
            time,
            inputs
    ):
        """Log probability under a categorical distribution.

        Args:
        - samples: samples from a categorical distribution
            with shape [batch_dim, 1]
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the categorical distribution
            with shape [batch_dim, inputs_dim[i]].

        Returns:
        - log_prob: the log probability of samples
            with shape [batch_dim].
        """
        log_probs = self.get_parameters(time, inputs)[0]
        return tf.gather_nd(
            log_probs, categorical_samples, batch_dims=1)
