"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


def create_shooting_update(
        controls_model,
        dynamics_model,
        cost_model
):
    """Predicts into the future using a policy and dynamics model.

    Args:
    - controls_model: the controls as a function.
        the function returns tensors with shape [batch_dim, controls_dim, 1].
    - dynamics_model: the dynamics as function.
        the model returns tensors with shape [batch_dim, state_dim, 1].
    - cost_model: the cost as a function.
        the function returns tensors with shape [batch_dim, 1, 1].

    Returns:
    - a function shooting_update
    """

    def shooting_update(
            x,
            time,
    ):
        """Predicts into the future using a policy and dynamics model.

        Args:
        - x: the current state with shape [batch_dim, state_dim, 1].
        - time: the current time step of shooting.

        Returns:
        - xp1: the next state with shape [batch_dim, state_dim, 1].
        - u: the controls with shape [batch_dim, controls_dim, 1].
        - c: the costs with shape [batch_dim, 1, 1].
        """
        u = controls_model(time, [x])
        xp1 = dynamics_model(time, [x, u])
        c = cost_model(time, [x, u])
        return xp1, u, c

    return shooting_update
