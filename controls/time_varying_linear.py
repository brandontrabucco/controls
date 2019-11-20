"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def time_varying_linear(
        origin_inputs,
        origin_outputs,
        jacobian
):
    """Create a function for a time varying linear model."""

    # get the batch shape and vector sizes

    batch_dim = tf.reduce_prod(tf.shape(jacobian)[1:-2])

    horizon = tf.shape(jacobian)[0]

    inputs_dim = jacobian.shape[-1]

    outputs_dim = jacobian.shape[-2]

    # create a time varying linear model

    origin_inputs = tf.reshape(
        origin_inputs, [horizon, batch_dim, inputs_dim, 1])

    origin_outputs = tf.reshape(
        origin_outputs, [horizon, batch_dim, outputs_dim, 1])

    jacobian = tf.reshape(
        jacobian, [horizon, batch_dim, outputs_dim, inputs_dim])

    time = -1

    def model(
            x
    ):
        """Compute a forward pass using the model."""

        nonlocal time

        time += 1

        tf.debugging.assert_less(
            time,
            horizon,
            message="cannot use model beyond original horizon")

        return origin_outputs[time, ...] + tf.matmul(
            jacobian[time, ...], x[0] - origin_inputs[time, ...])

    return model
