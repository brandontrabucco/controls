"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from abc import ABC, abstractmethod
import tensorflow as tf


class Distribution(ABC):

    def __init__(
            self,
            model
    ):
        """Create a distribution for a random variable.

        Args:
        - model: a function mapping time and inputs to parameters.
        """
        self.model = model

    def __call__(
            self,
            time,
            inputs
    ):
        """Returns the parameters of a random variable.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the distribution of a random variable
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - parameters: the parameters of the distribution.
        """
        return self.sample(time, inputs)[0]

    def get_parameters(
            self,
            time,
            inputs
    ):
        """Returns the parameters of a random variable.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the distribution of a random variable
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - parameters: the parameters of the distribution.
        """
        return self.model(time, inputs)

    @abstractmethod
    def sample(
            self,
            time,
            inputs
    ):
        """Sample from a random variable.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the distribution of a random variable
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - samples: samples from a distribution of a random variable
            with shape [batch_dim, outputs_dim, 1].
        - log_prob: the log probability of samples
            with shape [batch_dim, 1, 1].
        """
        return NotImplemented

    @abstractmethod
    def expected_value(
            self,
            time,
            inputs
    ):
        """Expectation of a random variable.

        Args:
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the distribution of a random variable
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - samples: samples from a distribution of a random variable
            with shape [batch_dim, outputs_dim, 1].
        - log_prob: the log probability of samples
            with shape [batch_dim, 1, 1].
        """
        return NotImplemented

    @abstractmethod
    def log_prob(
            self,
            samples,
            time,
            inputs
    ):
        """Expectation of a random variable.

        Args:
        - samples: samples from a distribution of a random variable
            with shape [batch_dim, outputs_dim, 1].
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the distribution of a random variable
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - log_prob: the log probability of samples
            with shape [batch_dim, 1, 1].
        """
        return NotImplemented

    def prob(
            self,
            samples,
            time,
            inputs
    ):
        """Expectation of a random variable.

        Args:
        - samples: samples from a distribution of a random variable
            with shape [batch_dim, outputs_dim, 1].
        - time: an integer representing the time step of the system.
        - inputs[i]: the inputs to the distribution of a random variable
            with shape [batch_dim, inputs_dim[i], 1].

        Returns:
        - prob: the probability density of samples
            with shape [batch_dim, 1, 1].
        """
        return tf.exp(self.log_prob(time, inputs))

    def __setattr__(
            self,
            attr,
            value
    ):
        if not attr == "model":
            setattr(self.model, attr, value)
        else:
            self.__dict__[attr] = value

    def __getattr__(
            self,
            attr
    ):
        if not attr == "model":
            return getattr(self.model, attr)
        else:
            return self.__dict__[attr]
