"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from diffopt.shooting import shooting
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
):
    """Solves for optimal controls using the cross entropy method.

    Args:
    - x0: the initial states from which to predict into the future
        with shape [batch_dim, state_dim].

    - controls_model: the initial policy as a random function.
        the function returns tensors with shape [batch_dim, controls_dim].
    - dynamics_model: the dynamics as a random function.
        the function returns tensors with shape [batch_dim, state_dim].
    - cost_model: the cost as a random function.
        the function returns tensors with shape [batch_dim, 1].

    - h: the number of steps into the future for the planner.
    - c: the number of candidates to samples.
    - n: the number of iterations to run cem over.
    - k: the number of samples to average.

    Returns:
    - controls_model: the policy as a random function.
        the function returns tensors with shape [batch_dim, controls_dim].
    """
    batch_dim = tf.shape(x0)[0]
    state_dim = tf.shape(x0)[1]

    # tile the states for the number of candidate runs
    x0 = tf.reshape(tf.tile(x0[:, tf.newaxis, :], [1, c, 1]), [batch_dim * c, state_dim])

    # iteratively run forward shooting and cross entropy minimization
    for iteration in range(n):

        # wrap the policy and tile the batch size by c
        if iteration > 0:
            original_model = controls_model.model
            controls_model.model = lambda time, inputs: [
                tf.reshape(tf.tile(x[:, tf.newaxis, ...], [1, c] + [1 for s in tf.shape(x)[1:]]),
                           [tf.shape(x)[0] * c] + [s for s in tf.shape(x)[1:]])
                for x in original_model(time, inputs)]

        # run an initial forward pass using the shooting algorithm
        xi, ui, ci = shooting(
            x0, controls_model, dynamics_model, cost_model, h=h, random=True)

        # compute the top k action samples by their negative cost
        best_costs, best_idx = tf.math.top_k(-tf.reduce_sum(
            tf.reshape(ci, [h, batch_dim, c]), axis=0), k=k)

        # infer the cardinality of the controls from the shooting
        controls_dim = tf.shape(ui)[2]

        # compute the diffopt sequences in the top k
        top_ui = tf.gather(tf.reshape(ui, [h, batch_dim, c, controls_dim]),
                           tf.tile(best_idx[tf.newaxis, :, :], [h, 1, 1]), axis=2, batch_dims=2)

        # update the distributions of the controls using the top k
        controls_model = controls_model.fit(top_ui)

    # return the latest and greatest controls model
    return controls_model
