"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.kf.kf import kf
import tensorflow as tf


if __name__ == "__main__":

    batch_dim = 10
    horizon = 20
    state_dim = 32
    measurement_dim = 24
    controls_dim = 7

    A = tf.eye(state_dim, batch_shape=[batch_dim, horizon])
    B = tf.random.normal([batch_dim, horizon, state_dim, controls_dim])
    C = tf.random.normal([batch_dim, horizon, measurement_dim, state_dim])

    states = [
        tf.random.normal([batch_dim, 1, state_dim, 1])]

    controls = [
        tf.random.normal([batch_dim, 1, controls_dim, 1])]

    measurements = [
        C[:, 0:1, :, :] @ states[-1] +
        tf.random.normal([batch_dim, 1, measurement_dim, 1]) * 0.01]

    for i in range(1, horizon):

        states.append(
            A[:, i:i+1, :, :] @ states[-1] +
            B[:, i:i+1, :, :] @ controls[-1] +
            tf.random.normal([batch_dim, 1, state_dim, 1]) * 0.01)

        controls.append(tf.random.normal([batch_dim, 1, controls_dim, 1]))

        measurements.append(
            C[:, i:i+1, :, :] @ states[-1] +
            tf.random.normal([batch_dim, 1, measurement_dim, 1]) * 0.01)

    states = tf.concat(states, 1)
    controls = tf.concat(controls, 1)
    measurements = tf.concat(measurements, 1)

    result = kf(
        measurements,
        tf.zeros([batch_dim, state_dim, 1]),
        tf.eye(state_dim, batch_shape=[batch_dim]),
        controls,
        A,
        B,
        tf.eye(state_dim, batch_shape=[batch_dim]) * 0.01,
        C,
        tf.eye(measurement_dim, batch_shape=[batch_dim]) * 0.01)

    error = tf.linalg.norm(result[0] - states, ord=1)
    print("error prediction", error.numpy())

    error = tf.linalg.norm(result[4] - states, ord=1)
    print("error filter", error.numpy())

    error = tf.linalg.norm(result[8] - states, ord=1)
    print("error smooth", error.numpy())