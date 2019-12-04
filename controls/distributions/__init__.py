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
