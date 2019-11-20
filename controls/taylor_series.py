"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def first_order(
        nonlinear_model,
        *inputs
):
    """Linearize a nonlinear vector model about a center.

    Args:
    - nonlinear_model: a nonlinear model that is a function
        the function accepts inputs of shape [batch_dim, input_dim, 1].
    - inputs[i]: the vector around which to build a local approximation
        with shape [batch_dim, input_dim, 1].

    Returns:
    - outputs: the outputs of the function at center
        with shape [batch_dim, output_dim, 1].
    - jacobians[i]: the jacobian of the outputs wrt. center
        with shape [batch_dim, output_dim, input_dim].
        """

    # check the shapes of the input tensor

    for i in range(len(inputs)):

        tf.debugging.assert_equal(
            3,
            tf.size(tf.shape(inputs[i])),
            message="inputs[{}] should be a 3 tensor".format(i))

        tf.debugging.assert_equal(
            1,
            tf.shape(inputs[i])[-1],
            message="inputs[{}] should have shape [batch_dim, input_dim, 1]".format(i))

    # approximate the model using gradients

    with tf.GradientTape(persistent=True) as tape:

        for i in range(len(inputs)):
            tape.watch(inputs[i])

        outputs = nonlinear_model(inputs)

        tf.debugging.assert_equal(
            3,
            tf.size(tf.shape(outputs)),
            message="outputs should be a 3 tensor")

        tf.debugging.assert_equal(
            1,
            tf.shape(outputs)[-1],
            message="outputs should have shape [batch_dim, output_dim, 1]")

    jacobians = [tape.batch_jacobian(
        outputs, inputs[i])[:, :, 0, :, 0] for i in range(len(inputs))]

    return (
        outputs,
        jacobians)


def second_order(
        nonlinear_model,
        *inputs
):
    """Quadratic approximate a nonlinear scalar model about a center.

    Args:
    - nonlinear_model: a nonlinear model that is a function
        the function accepts inputs of shape [batch_dim, input_dim, 1].
    - inputs[i]: the vector around which to build a local approximation
        with shape [batch_dim, input_dim, 1].

    Returns:
    - outputs: the outputs of the function at center
        with shape [batch_dim, 1, 1].
    - jacobians[i]: the jacobian of the outputs wrt. center
        with shape [batch_dim, input_dim, 1].
    - hessians[i]: the hessian of the outputs wrt. center
        with shape [batch_dim, input_dim, input_dim].
        """

    # check the shapes of the input tensor

    for i in range(len(inputs)):

        tf.debugging.assert_equal(
            3,
            tf.size(tf.shape(inputs[i])),
            message="inputs[{}] should be a 3 tensor".format(i))

        tf.debugging.assert_equal(
            1,
            tf.shape(inputs[i])[-1],
            message="inputs[{}] should have shape [batch_dim, input_dim, 1]".format(i))

    # approximate the model using gradients

    with tf.GradientTape(persistent=True) as outer_tape:

        for i in range(len(inputs)):
            outer_tape.watch(inputs[i])

        with tf.GradientTape(persistent=True) as inner_tape:

            for i in range(len(inputs)):
                inner_tape.watch(inputs[i])

            outputs = nonlinear_model(inputs)

            tf.debugging.assert_equal(
                3,
                tf.size(tf.shape(outputs)),
                message="outputs should be a 3 tensor")

            tf.debugging.assert_equal(
                1,
                tf.shape(outputs)[-1],
                message="outputs should have shape [batch_dim, 1, 1]")

            tf.debugging.assert_equal(
                1,
                tf.shape(outputs)[-2],
                message="outputs should have shape [batch_dim, 1, 1]")

        jacobians = [inner_tape.batch_jacobian(
            outputs, inputs[i])[:, 0, 0, :, :] for i in range(len(inputs))]

    hessians = [[outer_tape.batch_jacobian(
        jacobians[i], inputs[j], experimental_use_pfor=False)[:, :, 0, :, 0] for j in range(len(inputs))]
            for i in range(len(jacobians))]

    return (
        outputs,
        jacobians,
        hessians)
