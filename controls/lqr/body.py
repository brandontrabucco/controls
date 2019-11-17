"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.lqr.update import lqr_update


def lqr_body(
        controls_state_jacobian,
        value_hessian,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        cost_state_hessian,
        cost_controls_hessian,
        time,
        horizon
):
    """Inner body of lqr update loop."""

    # run the lqr and collect results

    update_result = lqr_update(
        value_hessian,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        cost_state_hessian,
        cost_controls_hessian)

    # propagate results through the bellman backup

    controls_state_jacobian = update_result[0]

    value_hessian = update_result[1]

    time = time + 1

    return (
        controls_state_jacobian,
        value_hessian,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        cost_state_hessian,
        cost_controls_hessian,
        time,
        horizon)
