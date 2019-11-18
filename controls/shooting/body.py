"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.shooting.update import shooting_update


def shooting_body(
        states,
        controls_state_jacobian,
        dynamics_model,
        states_array,
        controls_array,
        dynamics_state_jacobian_array,
        dynamics_controls_jacobian_array,
        time,
        horizon
):
    """Inner body of shooting update loop."""

    # run the lqr and collect results

    update_result = shooting_update(
        states,
        controls_state_jacobian[time, :, :, :],
        dynamics_model)

    # propagate results through the bellman backup

    predicted_states = update_result[0]

    controls = update_result[1]

    dynamics_state_jacobian = update_result[2]

    dynamics_controls_jacobian = update_result[3]

    states_array = states_array.write(
        time, states)

    controls_array = controls_array.write(
        time, controls)

    dynamics_state_jacobian_array = dynamics_state_jacobian_array.write(
        time, dynamics_state_jacobian)

    dynamics_controls_jacobian_array = dynamics_controls_jacobian_array.write(
        time, dynamics_controls_jacobian)

    time = time + 1

    return (
        predicted_states,
        controls_state_jacobian,
        dynamics_model,
        states_array,
        controls_array,
        dynamics_state_jacobian_array,
        dynamics_controls_jacobian_array,
        time,
        horizon)
