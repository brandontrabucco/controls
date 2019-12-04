"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.distributions.gaussian import Gaussian
from controls.distributions.tanh_gaussian import TanhGaussian
from controls.distributions.deterministic import Deterministic
from controls.distributions.categorical import Categorical
import tensorflow as tf


class UnitGaussian(Gaussian):

    def __init__(self, size=1):
        """Create a distribution for a unit gaussian variable.

        Args:
        - size: an integer, the size of the vector space.
        """
        Gaussian.__init__(
            self,
            lambda time, inputs: (
                tf.zeros([tf.shape(inputs[0])[0], size]),
                tf.ones([tf.shape(inputs[0])[0], size, size]),
                tf.ones([tf.shape(inputs[0])[0], size, size]),
                tf.zeros([tf.shape(inputs[0])[0]]),))


class UniformCategorical(Categorical):

    def __init__(self, size=1):
        """Create a distribution for a unit gaussian variable.

        Args:
        - size: the number of categorical options.
        """
        Categorical.__init__(
            self,
            lambda time, inputs: (
                tf.zeros([tf.shape(inputs[0])[0], size]),))


class LinearGaussian(Gaussian):

    def __init__(self, mean, covariance, jacobians):
        """Create a distribution for a linear gaussian.

        Args:
        - mean: a mean for the linear gaussian
            with shape [batch_dim, output_dim]
        - covariance: a covariance matrix for the linear gaussian
            with shape [batch_dim, output_dim, output_dim]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [batch_dim, output_dim, input_dim[i]]
        """
        def linear_gaussian_model(time, inputs):
            delta = 0.0

            for jacobian, x in zip(jacobians, inputs):
                if jacobian is not None and jacobian != 0:
                    delta = delta + (
                        jacobian @ x[:, :, tf.newaxis])[:, :, 0]

            return (delta + (mean if mean is not None else 0),
                    tf.linalg.sqrtm(covariance),
                    tf.linalg.inv(covariance),
                    tf.linalg.logdet(covariance))

        Gaussian.__init__(self, linear_gaussian_model)


class TimeVaryingLinearGaussian(Gaussian):

    def __init__(self, mean, covariance, jacobians):
        """Create a distribution for a time varying linear gaussian.

        Args:
        - mean: a mean for the linear gaussian
            with shape [T, batch_dim, output_dim]
        - covariance: a covariance matrix for the linear gaussian
            with shape [T, batch_dim, output_dim, output_dim]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [T, batch_dim, output_dim, input_dim[i]]
        """
        def linear_gaussian_model(time, inputs):
            delta = 0.0

            for jacobian, x in zip(jacobians, inputs):
                if jacobian is not None and jacobian != 0:
                    delta = delta + (
                        jacobian[time] @ x[:, :, tf.newaxis])[:, :, 0]

            return (delta + (mean[time] if mean is not None else 0),
                    tf.linalg.sqrtm(covariance[time]),
                    tf.linalg.inv(covariance[time]),
                    tf.linalg.logdet(covariance[time]))

        Gaussian.__init__(self, linear_gaussian_model)


class Linear(Deterministic):

    def __init__(self, center, jacobians):
        """Create a distribution for a linear variable.

        Args:
        - center: the center of the linear function
            with shape [batch_dim, output_dim]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [batch_dim, output_dim, input_dim[i]]
        """
        def linear_model(time, inputs):
            delta = 0.0

            for jacobian, x in zip(jacobians, inputs):
                if jacobian is not None and jacobian != 0:
                    delta = delta + (
                        jacobian @ x[:, :, tf.newaxis])[:, :, 0]

            return delta + (center if center is not None else 0)

        Deterministic.__init__(self, linear_model)


class TimeVaryingLinear(Deterministic):

    def __init__(self, center, jacobians):
        """Create a distribution for a time varying linear variable.

        Args:
        - center: the center of the linear function
            with shape [T, batch_dim, output_dim]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [T, batch_dim, output_dim, input_dim[i]]
        """
        def linear_model(time, inputs):
            delta = 0.0

            for jacobian, x in zip(jacobians, inputs):
                if jacobian is not None and jacobian != 0:
                    delta = delta + (
                        jacobian[time] @ x[:, :, tf.newaxis])[:, :, 0]

            return delta + (center[time] if center is not None else 0)

        Deterministic.__init__(self, linear_model)


class QuadraticGaussian(Gaussian):

    def __init__(self, mean, covariance, jacobians, hessians):
        """Create a distribution for a quadratic gaussian.

        Args:
        - mean: a mean for the quadratic gaussian
            with shape [batch_dim, output_dim]
        - covariance: a covariance matrix for the linear gaussian
            with shape [batch_dim, output_dim, output_dim]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [batch_dim, output_dim, input_dim[i]]
        - hessians[i][j]: a hessian of the output wrt. input[i] and input[j]
            with shape [batch_dim, output_dim, input_dim[i], input_dim[j]]
        """
        def linear_gaussian_model(time, inputs):
            delta = 0.0

            for jacobian, x in zip(jacobians, inputs):
                if jacobian is not None and jacobian != 0:
                    delta = delta + (
                        jacobian @ x[:, :, tf.newaxis])[:, :, 0]

            for tmp, x in zip(hessians, inputs):
                for hessian, y in zip(tmp, inputs):
                    if hessian is not None and hessian != 0:

                        out = tf.matmul(x[:, tf.newaxis, :, tf.newaxis],
                                        hessian, transpose_a=True)
                        out = tf.matmul(out, y[:, tf.newaxis, :, tf.newaxis])
                        delta = delta + 0.5 * out[:, :, 0, 0]

            return (delta + (mean if mean is not None else 0),
                    tf.linalg.sqrtm(covariance),
                    tf.linalg.inv(covariance),
                    tf.linalg.logdet(covariance))

        Gaussian.__init__(self, linear_gaussian_model)


class TimeVaryingQuadraticGaussian(Gaussian):

    def __init__(self, mean, covariance, jacobians, hessians):
        """Create a distribution for a time varying quadratic gaussian.

        Args:
        - mean: a mean for the linear gaussian
            with shape [T, batch_dim, output_dim]
        - covariance: a covariance matrix for the quadratic gaussian
            with shape [T, batch_dim, output_dim, output_dim]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [T, batch_dim, output_dim, input_dim[i]]
        - hessians[i][j]: a hessian of the output wrt. input[i] and input[j]
            with shape [batch_dim, output_dim, input_dim[i], input_dim[j]]
        """
        def quadratic_gaussian_model(time, inputs):
            delta = 0.0

            for jacobian, x in zip(jacobians, inputs):
                if jacobian is not None and jacobian != 0:
                    delta = delta + (
                        jacobian[time] @ x[:, :, tf.newaxis])[:, :, 0]

            for tmp, x in zip(hessians, inputs):
                for hessian, y in zip(tmp, inputs):
                    if hessian is not None and hessian != 0:

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

    def __init__(self, center, jacobians, hessians):
        """Create a distribution for a quadratic variable.

        Args:
        - center: the center of the quadratic function
            with shape [batch_dim, output_dim]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [batch_dim, output_dim, input_dim[i]]
        - hessians[i][j]: a hessian of the output wrt. input[i] and input[j]
            with shape [batch_dim, output_dim, input_dim[i], input_dim[j]]
        """
        def quadratic_model(time, inputs):
            delta = 0.0

            for jacobian, x in zip(jacobians, inputs):
                if jacobian is not None and jacobian != 0:
                    delta = delta + (
                        jacobian @ x[:, :, tf.newaxis])[:, :, 0]

            for tmp, x in zip(hessians, inputs):
                for hessian, y in zip(tmp, inputs):
                    if hessian is not None and hessian != 0:

                        out = tf.matmul(x[:, tf.newaxis, :, tf.newaxis],
                                        hessian, transpose_a=True)
                        out = tf.matmul(out, y[:, tf.newaxis, :, tf.newaxis])
                        delta = delta + 0.5 * out[:, :, 0, 0]

            return delta + (center if center is not None else 0)

        Deterministic.__init__(self, quadratic_model)


class TimeVaryingQuadratic(Deterministic):

    def __init__(self, center, jacobians, hessians):
        """Create a distribution for a time varying quadratic variable.

        Args:
        - center: the center of the quadratic function
            with shape [T, batch_dim, output_dim]
        - jacobians[i]: a jacobian of the output wrt. input[i]
            with shape [T, batch_dim, output_dim, input_dim[i]]
        - hessians[i][j]: a hessian of the output wrt. input[i] and input[j]
            with shape [batch_dim, output_dim, input_dim[i], input_dim[j]]
        """
        def quadratic_model(time, inputs):
            delta = 0.0

            for jacobian, x in zip(jacobians, inputs):
                if jacobian is not None and jacobian != 0:
                    delta = delta + (
                        jacobian[time] @ x[:, :, tf.newaxis])[:, :, 0]

            for tmp, x in zip(hessians, inputs):
                for hessian, y in zip(tmp, inputs):
                    if hessian is not None and hessian != 0:

                        out = tf.matmul(x[:, tf.newaxis, :, tf.newaxis],
                                        hessian[time], transpose_a=True)
                        out = tf.matmul(out, y[:, tf.newaxis, :, tf.newaxis])
                        delta = delta + 0.5 * out[:, :, 0, 0]

            return delta + (center[time] if center is not None else 0)

        Deterministic.__init__(self, quadratic_model)
