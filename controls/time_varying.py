"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def linear_model(
        origin_outputs,
        origin_inputs,
        jacobians
):
    """Create a function for a time varying linear model.

    Args:
    - origin_outputs: the outputs of the nonlinear function centered at origin_inputs.
        with shape [horizon, batch_dim, outputs_dim, 1].
    - origin_inputs[i]: the inputs that the taylor approximation is centered around.
        with shape [horizon, batch_dim, inputs_dim[i], 1].
    - jacobians[i]: the jacobian of the outputs with respect to inputs[i]
        with shape [horizon, batch_dim, outputs_dim, inputs_dim[i]].

    Returns:
    - model: a function representing a time varying linear approximation,
        which accepts inputs[i] with shape [horizon, batch_dim, inputs_dim[i], 1].
    """

    for i in range(len(origin_inputs)):

        # get the batch shape and vector sizes

        horizon = tf.shape(jacobians[i])[0]
        batch_dim = tf.shape(jacobians[i])[1]
        outputs_dim = jacobians[i].shape[2]
        inputs_dim = jacobians[i].shape[3]

        # create a time varying linear model

        origin_inputs[i] = tf.reshape(
            origin_inputs[i], [horizon, batch_dim, inputs_dim, 1])
        jacobians[i] = tf.reshape(
            jacobians[i], [horizon, batch_dim, outputs_dim, inputs_dim])

    origin_outputs = tf.reshape(
        origin_outputs, [horizon, batch_dim, outputs_dim, 1])

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

        result = origin_outputs[time, ...]

        for i in range(len(origin_inputs)):
            result = result + tf.matmul(
                jacobians[i][time, ...], x[i] - origin_inputs[i][time, ...])

        return result

    return model


def constant_model(
        outputs,
):
    """Create a function for a time varying constant model.

    Args:
    - outputs: the outputs of the nonlinear time varying function
        with shape [horizon, batch_dim, outputs_dim, 1].

    Returns:
    - model: a function representing a time varying function,
        which accepts inputs[i] with any shape.
    """

    tf.debugging.assert_less(
        4,
        tf.size(tf.shape(outputs)),
        message="outputs should be a 4 tensor")

    tf.debugging.assert_less(
        1,
        tf.shape(outputs)[3],
        message="outputs should have shape [horizon, batch_dim, outputs_dim, 1]")

    # get the batch shape and vector sizes

    horizon = tf.shape(outputs)[0]

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

        return outputs[time, ...]

    return model
