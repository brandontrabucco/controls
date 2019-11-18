"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.shooting.update import create_shooting_update


def create_shooting_body(
        controls_model,
        dynamics_model,
        cost_model,
):
    """Create Inner body of shooting update loop."""

    shooting_update = create_shooting_update(
        controls_model,
        dynamics_model,
        cost_model)

    def shooting_body(
            states,
            states_array,
            controls_array,
            costs_array,
            dynamics_state_jacobian_array,
            dynamics_controls_jacobian_array,
            dynamics_shift_array,
            cost_state_state_hessian_array,
            cost_state_controls_hessian_array,
            cost_controls_state_hessian_array,
            cost_controls_controls_hessian_array,
            cost_state_jacobian_array,
            cost_controls_jacobian_array,
            cost_shift_array,
            time,
            horizon
    ):
        """Inner body of shooting update loop."""

        # run the lqr and collect results

        update_result = shooting_update(states)

        # propagate results through the bellman backup

        predicted_states = update_result[0]

        states_array = states_array.write(
            time, states)

        controls_array = controls_array.write(
            time, update_result[1])

        costs_array = costs_array.write(
            time, update_result[2])

        dynamics_state_jacobian_array = dynamics_state_jacobian_array.write(
            time, update_result[3])

        dynamics_controls_jacobian_array = dynamics_controls_jacobian_array.write(
            time, update_result[4])

        dynamics_shift_array = dynamics_shift_array.write(
            time, update_result[5])

        cost_state_state_hessian_array = cost_state_state_hessian_array.write(
            time, update_result[6])

        cost_state_controls_hessian_array = cost_state_controls_hessian_array.write(
            time, update_result[7])

        cost_controls_state_hessian_array = cost_controls_state_hessian_array.write(
            time, update_result[8])

        cost_controls_controls_hessian_array = cost_controls_controls_hessian_array.write(
            time, update_result[9])

        cost_state_jacobian_array = cost_state_jacobian_array.write(
            time, update_result[10])

        cost_controls_jacobian_array = cost_controls_jacobian_array.write(
            time, update_result[11])

        cost_shift_array = cost_shift_array.write(
            time, update_result[12])

        time = time + 1

        return (
            predicted_states,
            states_array,
            controls_array,
            costs_array,
            dynamics_state_jacobian_array,
            dynamics_controls_jacobian_array,
            dynamics_shift_array,
            cost_state_state_hessian_array,
            cost_state_controls_hessian_array,
            cost_controls_state_hessian_array,
            cost_controls_controls_hessian_array,
            cost_state_jacobian_array,
            cost_controls_jacobian_array,
            cost_shift_array,
            time,
            horizon)

    return shooting_body
