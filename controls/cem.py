"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.shooting.shooting import shooting
from controls.distributions.deterministic import Deterministic
import tensorflow as tf


def cem(
        x0,
        controls_model,
        dynamics_model,
        cost_model,
        h=10,
        c=1000,
        n=10,
        k=100,
        s=0.1
):
    """Solves for optimal actions using the cross entropy method.

    Args:
    - x0: the initial states from which to predict into the future
        with shape [batch_dim, state_dim, 1].

    - controls_model: the initial policy as a distribution.
        the function returns tensors with shape [batch_dim, controls_dim, 1].
    - dynamics_model: the dynamics as a distribution.
        the function returns tensors with shape [batch_dim, state_dim, 1].
    - cost_model: the cost as a distribution.
        the function returns tensors with shape [batch_dim, 1, 1].

    - h: the number of steps into the future for the planner.
    - c: the number of candidates to samples.
    - n: the number of iterations to run cem over.
    - k: the number of samples to average.
    - s: the std of noise to apply to the action samples.

    Returns:
    - controls_model: the policy as a function.
        the function returns tensors with shape [batch_dim, controls_dim, 1].
    """
    batch_dim = tf.shape(x0)[0]
    dtype = x0.dtype

    # tile the states for the number of candidate runs
    x0 = tf.tile(x0, [c, 1, 1])

    # iteratively run forward shooting and backward controls optimization with lqr
    for iteration in range(n):

        # wrap the controls model with exploration noise
        def wrapped_controls_model(time, inputs):
            wrapped_ui = tf.tile(controls_model(time, inputs)[:, tf.newaxis, :, :], [1, c, 1, 1])
            wrapped_ui = tf.reshape(wrapped_ui, [batch_dim * c, tf.shape(wrapped_ui)[2], 1])
            return wrapped_ui + s * tf.random.normal(tf.shape(wrapped_ui), dtype=dtype)

        # run an initial forward pass using the shooting algorithm
        inner_controls_model = Deterministic(wrapped_controls_model)
        xi, ui, ci = shooting(
            x0, inner_controls_model, dynamics_model, cost_model, h)

        # compute the top k action samples by their negative cost
        best_idx = tf.math.top_k(-tf.reduce_sum(
            tf.reshape(ci, [h, batch_dim, c]), axis=0), k=k)[1]

        # infer the cardinality of the controls from the shooting
        controls_dim = tf.shape(ui)[2]

        # compute the controls sequences in the top k
        top_ui = tf.gather(tf.reshape(ui, [h, batch_dim, c, controls_dim, 1]),
                           tf.tile(best_idx[tf.newaxis, :, :], [h, 1, 1]), axis=2, batch_dims=2)

        # compute the empirical average of the best candidate controls
        ui = tf.reduce_mean(top_ui, axis=2)
        controls_model = Deterministic(lambda time, inputs: ui[time, :, :, :])

    # return the latest and greatest controls model
    return controls_model
