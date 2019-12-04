"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.distributions.continuous.gaussian import Gaussian
from controls.distributions.continuous.deterministic import Deterministic
import tensorflow as tf


class QuadraticGaussian(Gaussian):

    def __init__(self, mean, centers, covariance, jacobians, hessians):
        """Create a distribution for a quadratic gaussian.

        Args:
        - mean: a mean for the quadratic gaussian
            with shape [batch_dim, output_dim]
        - centers[i]: the center of the taylor approximation
            with shape [batch_dim, input_dim[i]]
        - covariance: a covariance matrix for the linear gaussian
            with shape [batch_dim, output_dim, output_dim]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [batch_dim, output_dim, input_dim[i]]
        - hessians[i][j]: a hessian of the output wrt. input[i] and input[j]
            with shape [batch_dim, output_dim, input_dim[i], input_dim[j]]
        """
        def quadratic_gaussian_model(time, inputs):
            delta = 0.0

            for jacobian, x, x0 in zip(jacobians, inputs, centers):
                x = x - (x0 if x0 is not None else 0)

                if jacobian is not None:
                    delta = delta + (
                        jacobian @ x[:, :, tf.newaxis])[:, :, 0]

            for tmp, x, x0 in zip(hessians, inputs, centers):
                x = x - (x0 if x0 is not None else 0)

                for hessian, y, y0 in zip(tmp, inputs, centers):
                    y = y - (y0 if y0 is not None else 0)

                    if hessian is not None:
                        out = tf.matmul(x[:, tf.newaxis, :, tf.newaxis],
                                        hessian, transpose_a=True)
                        out = tf.matmul(out, y[:, tf.newaxis, :, tf.newaxis])
                        delta = delta + 0.5 * out[:, :, 0, 0]

            return (delta + (mean if mean is not None else 0),
                    tf.linalg.sqrtm(covariance),
                    tf.linalg.inv(covariance),
                    tf.linalg.logdet(covariance))

        Gaussian.__init__(self, quadratic_gaussian_model)


class TimeVaryingQuadraticGaussian(Gaussian):

    def __init__(self, mean, centers, covariance, jacobians, hessians):
        """Create a distribution for a time varying quadratic gaussian.

        Args:
        - mean: a mean for the linear gaussian
            with shape [T, batch_dim, output_dim]
        - centers[i]: the center of the taylor approximation
            with shape [T, batch_dim, input_dim[i]]
        - covariance: a covariance matrix for the quadratic gaussian
            with shape [T, batch_dim, output_dim, output_dim]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [T, batch_dim, output_dim, input_dim[i]]
        - hessians[i][j]: a hessian of the output wrt. input[i] and input[j]
            with shape [batch_dim, output_dim, input_dim[i], input_dim[j]]
        """
        def quadratic_gaussian_model(time, inputs):
            delta = 0.0

            for jacobian, x, x0 in zip(jacobians, inputs, centers):
                x = x - (x0[time] if x0 is not None else 0)

                if jacobian is not None:
                    delta = delta + (
                        jacobian[time] @ x[:, :, tf.newaxis])[:, :, 0]

            for tmp, x, x0 in zip(hessians, inputs, centers):
                x = x - (x0[time] if x0 is not None else 0)

                for hessian, y, y0 in zip(tmp, inputs, centers):
                    y = y - (y0[time] if y0 is not None else 0)

                    if hessian is not None:
                        out = tf.matmul(x[:, tf.newaxis, :, tf.newaxis],
                                        hessian, transpose_a=True)
                        out = tf.matmul(out, y[:, tf.newaxis, :, tf.newaxis])
                        delta = delta + 0.5 * out[:, :, 0, 0]

            return (delta + (mean[time] if mean is not None else 0),
                    tf.linalg.sqrtm(covariance[time]),
                    tf.linalg.inv(covariance[time]),
                    tf.linalg.logdet(covariance[time]))

        Gaussian.__init__(self, quadratic_gaussian_model)


class Quadratic(Deterministic):

    def __init__(self, mean, centers, jacobians, hessians):
        """Create a distribution for a quadratic variable.

        Args:
        - mean: the mean of the quadratic function
            with shape [batch_dim, output_dim]
        - centers[i]: the center of the taylor approximation
            with shape [batch_dim, input_dim[i]]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [batch_dim, output_dim, input_dim[i]]
        - hessians[i][j]: a hessian of the output wrt. input[i] and input[j]
            with shape [batch_dim, output_dim, input_dim[i], input_dim[j]]
        """
        def quadratic_model(time, inputs):
            delta = 0.0

            for jacobian, x, x0 in zip(jacobians, inputs, centers):
                x = x - (x0 if x0 is not None else 0)

                if jacobian is not None:
                    delta = delta + (
                        jacobian @ x[:, :, tf.newaxis])[:, :, 0]

            for tmp, x, x0 in zip(hessians, inputs, centers):
                x = x - (x0 if x0 is not None else 0)

                for hessian, y, y0 in zip(tmp, inputs, centers):
                    y = y - (y0 if y0 is not None else 0)

                    if hessian is not None:
                        out = tf.matmul(x[:, tf.newaxis, :, tf.newaxis],
                                        hessian, transpose_a=True)
                        out = tf.matmul(out, y[:, tf.newaxis, :, tf.newaxis])
                        delta = delta + 0.5 * out[:, :, 0, 0]

            return delta + (mean if mean is not None else 0),

        Deterministic.__init__(self, quadratic_model)


class TimeVaryingQuadratic(Deterministic):

    def __init__(self, mean, centers, jacobians, hessians):
        """Create a distribution for a time varying quadratic variable.

        Args:
        - mean: the mean of the quadratic function
            with shape [T, batch_dim, output_dim]
        - centers[i]: the center of the taylor approximation
            with shape [T, batch_dim, input_dim[i]]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [T, batch_dim, output_dim, input_dim[i]]
        - hessians[i][j]: a hessian of the output wrt. input[i] and input[j]
            with shape [batch_dim, output_dim, input_dim[i], input_dim[j]]
        """
        def quadratic_model(time, inputs):
            delta = 0.0

            for jacobian, x, x0 in zip(jacobians, inputs, centers):
                x = x - (x0[time] if x0 is not None else 0)

                if jacobian is not None:
                    delta = delta + (
                        jacobian[time] @ x[:, :, tf.newaxis])[:, :, 0]

            for tmp, x, x0 in zip(hessians, inputs, centers):
                x = x - (x0[time] if x0 is not None else 0)

                for hessian, y, y0 in zip(tmp, inputs, centers):
                    y = y - (y0[time] if y0 is not None else 0)

                    if hessian is not None:
                        out = tf.matmul(x[:, tf.newaxis, :, tf.newaxis],
                                        hessian[time], transpose_a=True)
                        out = tf.matmul(out, y[:, tf.newaxis, :, tf.newaxis])
                        delta = delta + 0.5 * out[:, :, 0, 0]

            return delta + (mean[time] if mean is not None else 0),

        Deterministic.__init__(self, quadratic_model)
