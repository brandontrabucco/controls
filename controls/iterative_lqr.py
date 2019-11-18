"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


def iterative_lqr(
        initial_states,
        dynamics_model,
        horizon,
        cost_state_hessian,
        cost_controls_hessian
):
    """Solves for the value iteration solution to lqr iteratively.

    Args:
    - initial_states: the initial states from which to predict into the future
        with shape [batch_dim, ..., state_dim, 1].
    - dynamics_model: the dynamics as a tensorflow.keras.Model.
    - horizon: the number of time steps for planning into the future.

    Returns:
    - predicted_states: the next state with shape [T, batch_dim, ..., state_dim, 1].
    - controls: the controls with shape [batch_dim, controls_dim, 1].
    - dynamics_state_jacobian: the jacobian of the dynamics wrt. the state
        with shape [T, batch_dim, ..., state_dim, state_dim].
    - dynamics_controls_jacobian: the jacobian of the dynamics wrt. the controls
        with shape [T, batch_dim, ..., state_dim, controls_dim].
        """
    pass