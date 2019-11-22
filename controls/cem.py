"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.shooting.shooting import shooting
from controls.time_varying import constant_model
import tensorflow as tf


def cem(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        horizon=10,
        num_candidates=1000,
        num_iterations=10,
        top_k=100,
        exploration_noise_std=0.1
):
    """Solves for optimal actions using the cross entropy method.

    Args:
    - initial_states: the initial states from which to predict into the future
        with shape [batch_dim, state_dim, 1].

    - controls_model: the initial policy as a function.
        the function returns tensors with shape [batch_dim, controls_dim, 1].
    - dynamics_model: the dynamics as a function.
        the function returns tensors with shape [batch_dim, state_dim, 1].
    - cost_model: the cost as a function.
        the function returns tensors with shape [batch_dim, 1, 1].

    - horizon: the number of steps into the future for the planner.
    - num_candidates: the number of candidates to samples.
    - num_iterations: the number of iterations to run cem over.
    - top_k: the number of samples to average.
    - exploration_noise_std: the std of noise to apply to the action samples.

    Returns:
    - controls_model: the policy as a function.
        the function returns tensors with shape [batch_dim, controls_dim, 1].
    """

    # check that initial_states is a 3 tensor
    tf.debugging.assert_equal(
        3,
        tf.size(tf.shape(initial_states)),
        message="initial_states should be a 3 tensor")

    # get the batch shape and vector sizes
    batch_dim = tf.shape(initial_states)[0]
    dtype = initial_states.dtype

    # tile the states for the number of candidate runs
    initial_states = tf.tile(initial_states, [num_candidates, 1, 1])

    # iteratively run forward shooting and backward controls optimization with lqr
    for i in range(num_iterations):

        # wrap the previous controls model with a gaussian random exploration
        def exploration_controls_model(x):
            wrapped_controls = controls_model(x)
            return wrapped_controls + exploration_noise_std * tf.random.normal(
                tf.shape(wrapped_controls), dtype=dtype)

        # run an initial forward pass using the shooting algorithm
        candidate_states, candidate_controls, candidate_costs = shooting(
            initial_states, exploration_controls_model, dynamics_model, cost_model, horizon)

        # compute the top k action samples by their negative cost
        best_idx = tf.math.top_k(-tf.reduce_sum(
            tf.reshape(candidate_costs, [horizon, batch_dim, num_candidates]), axis=0), k=top_k)[1]

        # infer the cardinality of the controls from the shooting
        controls_dim = tf.shape(candidate_controls)[2]

        # compute the controls sequences in the top k
        best_controls = tf.gather(
            tf.reshape(candidate_controls, [horizon, batch_dim, num_candidates, controls_dim, 1]),
            tf.tile(best_idx[tf.newaxis, :, :], [horizon, 1, 1]), axis=2, batch_dims=2)

        # compute the empirical average of the best candidate controls
        controls_model = constant_model(tf.tile(tf.reduce_mean(
            best_controls, axis=2), [1, num_candidates if i < num_iterations - 1 else 1, 1, 1]))

    # return the latest and greatest controls model
    return controls_model
