"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls import iterative_lqr
from controls import shooting
import tensorflow as tf

from controls.distributions.deterministic import Deterministic

if __name__ == "__main__":

    size = 5

    goal = tf.ones([1, size, 1])

    controls_model = Deterministic(lambda time, inputs: tf.zeros([1, size, 1]))
    dynamics_model = Deterministic(lambda time, inputs: inputs[0] + inputs[1])
    cost_model = Deterministic(lambda time, inputs: tf.matmul(
        inputs[0] - goal, inputs[0] - goal, transpose_a=True))

    initial_states = -tf.ones([1, size, 1])

    controls_model = iterative_lqr(
        initial_states,
        controls_model,
        dynamics_model,
        cost_model,
        h=20,
        n=10,
        a=0.1)

    shooting_states, shooting_controls, shooting_costs = shooting(
        initial_states, controls_model, dynamics_model, cost_model, h=20)

    for i in range(20):

        costs = shooting_costs[i, ...]

        print("Cost: {}".format(costs.numpy().sum()))
