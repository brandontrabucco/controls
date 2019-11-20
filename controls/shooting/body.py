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

        time = time + 1

        return (
            predicted_states,
            states_array,
            controls_array,
            costs_array,
            time,
            horizon)

    return shooting_body
