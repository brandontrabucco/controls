"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from diffopt import Deterministic
from diffopt import iterative_lqr
from diffopt import shooting
import tensorflow as tf


if __name__ == "__main__":

    size = 5
    goal = tf.ones([1, size])
    initial_states = -tf.ones([1, size])

    controls_model = Deterministic(lambda time, inputs: [tf.zeros([1, size])])
    dynamics_model = Deterministic(lambda time, inputs: [inputs[0] + inputs[1]])
    cost_model = Deterministic(lambda time, inputs: [tf.reduce_sum(
        (inputs[0] - goal)**2, axis=1, keepdims=True)])

    controls_model = iterative_lqr(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        h=20,
        n=10,
        a=0.01,
        random=False)

    shooting_states, shooting_controls, shooting_costs = shooting(
        initial_states, controls_model, dynamics_model, cost_model, h=20, random=False)

    for i in range(20):

        costs = shooting_costs[i, ...]

        print("Cost: {}".format(costs.numpy().sum()))
