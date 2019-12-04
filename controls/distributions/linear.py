"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.distributions.continuous.gaussian import Gaussian
from controls.distributions.continuous.deterministic import Deterministic
import tensorflow as tf


class LinearGaussian(Gaussian):

    def __init__(self, mean, centers, covariance, jacobians):
        """Create a distribution for a linear gaussian.

        Args:
        - mean: a mean for the linear gaussian
            with shape [batch_dim, output_dim]
        - centers[i]: the center of the taylor approximation
            with shape [batch_dim, input_dim[i]]
        - covariance: a covariance matrix for the linear gaussian
            with shape [batch_dim, output_dim, output_dim]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [batch_dim, output_dim, input_dim[i]]
        """
        def linear_gaussian_model(time, inputs):
            delta = 0.0

            for jacobian, x, x0 in zip(jacobians, inputs, centers):
                x = x - (x0 if x0 is not None else 0)

                if jacobian is not None and jacobian != 0:
                    delta = delta + (
                        jacobian @ x[:, :, tf.newaxis])[:, :, 0]

            return (delta + (mean if mean is not None else 0),
                    tf.linalg.sqrtm(covariance),
                    tf.linalg.inv(covariance),
                    tf.linalg.logdet(covariance))

        Gaussian.__init__(self, linear_gaussian_model)


class TimeVaryingLinearGaussian(Gaussian):

    def __init__(self, mean, centers, covariance, jacobians):
        """Create a distribution for a time varying linear gaussian.

        Args:
        - mean: a mean for the linear gaussian
            with shape [T, batch_dim, output_dim]
        - centers[i]: the center of the taylor approximation
            with shape [T, batch_dim, input_dim[i]]
        - covariance: a covariance matrix for the linear gaussian
            with shape [T, batch_dim, output_dim, output_dim]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [T, batch_dim, output_dim, input_dim[i]]
        """
        def linear_gaussian_model(time, inputs):
            delta = 0.0

            for jacobian, x, x0 in zip(jacobians, inputs, centers):
                x = x - (x0[time] if x0 is not None else 0)

                if jacobian is not None and jacobian != 0:
                    delta = delta + (
                        jacobian[time] @ x[:, :, tf.newaxis])[:, :, 0]

            return (delta + (mean[time] if mean is not None else 0),
                    tf.linalg.sqrtm(covariance[time]),
                    tf.linalg.inv(covariance[time]),
                    tf.linalg.logdet(covariance[time]))

        Gaussian.__init__(self, linear_gaussian_model)


class Linear(Deterministic):

    def __init__(self, mean, centers, jacobians):
        """Create a distribution for a linear variable.

        Args:
        - mean: the mean of the linear function
            with shape [batch_dim, output_dim]
        - centers[i]: the center of the taylor approximation
            with shape [batch_dim, input_dim[i]]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [batch_dim, output_dim, input_dim[i]]
        """
        def linear_model(time, inputs):
            delta = 0.0

            for jacobian, x, x0 in zip(jacobians, inputs, centers):
                x = x - (x0 if x0 is not None else 0)

                if jacobian is not None and jacobian != 0:
                    delta = delta + (
                        jacobian @ x[:, :, tf.newaxis])[:, :, 0]

            return delta + (mean if mean is not None else 0)

        Deterministic.__init__(self, linear_model)


class TimeVaryingLinear(Deterministic):

    def __init__(self, mean, centers, jacobians):
        """Create a distribution for a time varying linear variable.

        Args:
        - mean: the mean of the linear function
            with shape [T, batch_dim, output_dim]
        - centers[i]: the center of the taylor approximation
            with shape [T, batch_dim, input_dim[i]]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [T, batch_dim, output_dim, input_dim[i]]
        """
        def linear_model(time, inputs):
            delta = 0.0

            for jacobian, x, x0 in zip(jacobians, inputs, centers):
                x = x - (x0[time] if x0 is not None else 0)

                if jacobian is not None and jacobian != 0:
                    delta = delta + (
                        jacobian[time] @ x[:, :, tf.newaxis])[:, :, 0]

            return delta + (mean[time] if mean is not None else 0)

        Deterministic.__init__(self, linear_model)
