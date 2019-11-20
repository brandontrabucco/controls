"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.lqr.update import lqr_update


def lqr_body(
        value_state_state_hessian,
        value_state_jacobian,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        dynamics_shift,
        cost_state_state_hessian,
        cost_state_controls_hessian,
        cost_controls_state_hessian,
        cost_controls_controls_hessian,
        cost_state_jacobian,
        cost_controls_jacobian,
        controls_state_jacobian_array,
        controls_shift_array,
        value_state_state_hessian_array,
        value_state_jacobian_array,
        time,
        horizon
):
    """Inner body of lqr update loop."""

    # run the lqr and collect results

    update_result = lqr_update(
        value_state_state_hessian,
        value_state_jacobian,
        dynamics_state_jacobian[time, :, :, :],
        dynamics_controls_jacobian[time, :, :, :],
        dynamics_shift[time, :, :, :],
        cost_state_state_hessian[time, :, :, :],
        cost_state_controls_hessian[time, :, :, :],
        cost_controls_state_hessian[time, :, :, :],
        cost_controls_controls_hessian[time, :, :, :],
        cost_state_jacobian[time, :, :, :],
        cost_controls_jacobian[time, :, :, :])

    # propagate results through the bellman backup

    controls_state_jacobian = update_result[0]

    controls_shift = update_result[1]

    value_state_state_hessian = update_result[2]

    value_state_jacobian = update_result[3]

    controls_state_jacobian_array = controls_state_jacobian_array.write(
        time, controls_state_jacobian)

    controls_shift_array = controls_shift_array.write(
        time, controls_shift)

    value_state_state_hessian_array = value_state_state_hessian_array.write(
        time, value_state_state_hessian)

    value_state_jacobian_array = value_state_jacobian_array.write(
        time, value_state_jacobian)

    time = time - 1

    return (
        value_state_state_hessian,
        value_state_jacobian,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        dynamics_shift,
        cost_state_state_hessian,
        cost_state_controls_hessian,
        cost_controls_state_hessian,
        cost_controls_controls_hessian,
        cost_state_jacobian,
        cost_controls_jacobian,
        controls_state_jacobian_array,
        controls_shift_array,
        value_state_state_hessian_array,
        value_state_jacobian_array,
        time,
        horizon)
