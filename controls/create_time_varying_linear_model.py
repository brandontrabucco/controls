"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def create_time_varying_linear_model(
        inputs,
        outputs,
        jacobian,
        shift,
):
    """Create a keras model for a time varying linear taylor approximation."""

    # get the batch shape and vector sizes

    batch_shape = tf.shape(inputs)[1:-2]

    batch_dim = tf.reduce_prod(batch_shape)

    horizon = tf.shape(jacobian)[0]

    inputs_dim = jacobian.shape[-1]

    outputs_dim = jacobian.shape[-2]

    # create a time varying linear model

    inputs = tf.reshape(
        inputs, [horizon, batch_dim, inputs_dim, 1])

    outputs = tf.reshape(
        outputs, [horizon, batch_dim, outputs_dim, 1])

    jacobian = tf.reshape(
        jacobian, [horizon, batch_dim, outputs_dim, inputs_dim])

    shift = tf.reshape(
        shift, [horizon, batch_dim, outputs_dim, 1])

    time = -2

    def inner_model(
            x
    ):
        """Compute a forward pass using the model."""

        nonlocal time

        time += 1

        tf.debugging.assert_less(
            time,
            horizon,
            message="cannot use time varying linear model beyond horizon")

        return outputs[time, ...] + shift[time, ...] + tf.matmul(
            jacobian[time, ...], x[0] - inputs[time, ...])

    return inner_model
