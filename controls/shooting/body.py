"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.shooting.update import create_shooting_update


def create_shooting_body(
        controls_model,
        dynamics_model,
        cost_model,
):
    shooting_update = create_shooting_update(
        controls_model,
        dynamics_model,
        cost_model)

    def shooting_body(
        x,
        x_array,
        u_array,
        c_array,
        time,
        horizon
    ):
        xp1, u, c = shooting_update(x, time)
        return (
            xp1,
            x_array.write(time, x),
            u_array.write(time, u),
            c_array.write(time, c),
            time + 1,
            horizon)

    return shooting_body
