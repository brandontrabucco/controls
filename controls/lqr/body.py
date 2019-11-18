"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.lqr.update import lqr_update


def lqr_body(
        controls_state_jacobian,
        value_hessian,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        cost_state_hessian,
        cost_controls_hessian,
        controls_state_jacobian_array,
        value_hessian_array,
        time,
        horizon
):
    """Inner body of lqr update loop."""

    # run the lqr and collect results

    update_result = lqr_update(
        value_hessian,
        dynamics_state_jacobian[time, :, :, :],
        dynamics_controls_jacobian[time, :, :, :],
        cost_state_hessian[time, :, :, :],
        cost_controls_hessian[time, :, :, :])

    # propagate results through the bellman backup

    controls_state_jacobian = update_result[0]

    value_hessian = update_result[1]

    controls_state_jacobian_array = controls_state_jacobian_array.write(
        time, controls_state_jacobian)

    value_hessian_array = value_hessian_array.write(
        time, value_hessian)

    time = time - 1

    return (
        controls_state_jacobian,
        value_hessian,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        cost_state_hessian,
        cost_controls_hessian,
        controls_state_jacobian_array,
        value_hessian_array,
        time,
        horizon)
