"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def kf_smooth(
    mean,
    covariance,
    next_predicted_mean,
    next_predicted_covariance,
    next_smooth_mean,
    next_smooth_covariance,
    dynamics_state_jacobian,
):
    """Compute smooth estimates from a kalman filter.

    Args:
    - mean: the mean of the state estimate with shape [batch_dim, state_dim, 1].
    - covariance: the covariance of the state estimate with shape [batch_dim, state_dim, state_dim].

    - next_predicted_mean: the predicted mean of the next state with shape [batch_dim, state_dim, 1].
    - next_predicted_covariance: the predicted covariance of the next state with shape [batch_dim, state_dim, state_dim].

    - next_smooth_mean: the smooth mean of the next state with shape [batch_dim, state_dim, 1].
    - next_smooth_covariance: the smooth covariance of the next state with shape [batch_dim, state_dim, state_dim].

    - dynamics_state_jacobian: the jacobian of the dynamics wrt. the previous state.

    Returns:
    - smooth_gain: the gain used to compute smoothed values.
    - smooth_mean: the predicted next state without measurement.
    - smooth_covariance: the covariance of the predicted next state.
        """

    # get the batch shape and vector sizes

    batch_dim = tf.shape(mean)[0]

    # make sure all inputs conform to the batch shape

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(mean)[:-2],
        message="mean should have scalar batch shape")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(covariance)[:-2],
        message="covariance should have same batch size as mean")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(next_predicted_mean)[:-2],
        message="next predicted mean should have same batch size as mean")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(next_predicted_covariance)[:-2],
        message="next predicted covariance should have same batch size as mean")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(next_smooth_mean)[:-2],
        message="next smooth mean should have same batch size as mean")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(next_smooth_covariance)[:-2],
        message="next smooth covariance should have same batch size as mean")

    # compute the smoothing gain

    smooth_gain = tf.matmul(
        tf.matmul(
            covariance,
            dynamics_state_jacobian,
            transpose_b=True), tf.linalg.pinv(next_predicted_covariance))

    # compute the smoothed estimates

    smooth_mean = mean + tf.matmul(
        smooth_gain,
        next_smooth_mean - next_predicted_mean)

    smooth_covariance = covariance + tf.matmul(
        smooth_gain,
        tf.matmul(
            next_smooth_covariance - next_predicted_covariance, smooth_gain, transpose_b=True))

    return (
        smooth_gain,
        smooth_mean,
        smooth_covariance)


def kf_smooth_loop_body(
        mean,
        covariance,
        predicted_mean,
        predicted_covariance,
        next_smooth_mean,
        next_smooth_covariance,
        dynamics_state_jacobian,
        smooth_gain_array,
        smooth_mean_array,
        smooth_covariance_array,
        time
):
    """Inner body of kalman filter smoothing update loop."""

    # run the kalman filter smoothing step and collect results

    update_result = kf_smooth(
        mean[:, time, :, :],
        covariance[:, time, :, :],
        predicted_mean[:, time + 1, :, :],
        predicted_covariance[:, time + 1, :, :],
        next_smooth_mean,
        next_smooth_covariance,
        dynamics_state_jacobian[:, time, :, :])

    # push the results into arrays for safe keeping

    next_smooth_mean = update_result[1]

    next_smooth_covariance = update_result[2]

    smooth_gain_array = smooth_gain_array.write(
        time, update_result[0])

    smooth_mean_array = smooth_mean_array.write(
        time, next_smooth_mean)

    smooth_covariance_array = smooth_covariance_array.write(
        time, next_smooth_covariance)

    time = time - 1

    return (
        mean,
        covariance,
        predicted_mean,
        predicted_covariance,
        next_smooth_mean,
        next_smooth_covariance,
        dynamics_state_jacobian,
        smooth_gain_array,
        smooth_mean_array,
        smooth_covariance_array,
        time)


def kf_smooth_loop_condition(
        mean,
        covariance,
        predicted_mean,
        predicted_covariance,
        next_smooth_mean,
        next_smooth_covariance,
        dynamics_state_jacobian,
        smooth_gain_array,
        smooth_mean_array,
        smooth_covariance_array,
        time
):
    """Inner condition of kalman filter smooth update loop."""

    # exit the loop if time is less than zero

    return tf.greater(time, 0)
