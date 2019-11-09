"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


import tensorflow as tf


def kf_em(
    measurement,
    mean,
    covariance,
    controls,
    next_predicted_mean,
    next_predicted_covariance,
    smooth_gain,
    smooth_mean,
    smooth_covariance,
    next_smooth_mean,
    next_smooth_covariance,
    dynamics_state_jacobian,
    dynamics_controls_jacobian,
    measurement_jacobian,
):
    """Compute smooth estimates from a kalman filter.

    Args:
    - measurement: a vector of measurements with shape [batch_dim, measurement_dim, 1].
    - mean: the mean of the state estimate with shape [batch_dim, state_dim, 1].
    - covariance: the covariance of the state estimate with shape [batch_dim, state_dim, state_dim].
    - controls: the controls that modify the state with shape [batch_dim, controls_dim, 1].

    - next_predicted_mean: the predicted mean of the next state with shape [batch_dim, state_dim, 1].
    - next_predicted_covariance: the predicted covariance of the next state with shape [batch_dim, state_dim, state_dim].

    - smooth_gain: the matrix gain used for smoothing with shape [batch_dim, state_dim, state_dim].
    - smooth_mean: the smooth mean of the state with shape [batch_dim, state_dim, 1].
    - smooth_covariance: the smooth covariance of the state with shape [batch_dim, state_dim, state_dim].

    - next_smooth_mean: the smooth mean of the next state with shape [batch_dim, state_dim, 1].
    - next_smooth_covariance: the smooth covariance of the next state with shape [batch_dim, state_dim, state_dim].

    - dynamics_state_jacobian: the jacobian of the dynamics wrt. the previous state with shape [batch_dim, state_dim, state_dim].
    - dynamics_controls_jacobian: the jacobian of the dynamics wrt. the controls with shape [batch_dim, state_dim, controls_dim].

    - measurement_jacobian: the jacobian of the measurement wrt. the state with shape [batch_dim, measurement_dim, state_dim].

    Returns:
    - dynamics_covariance: the maximum likelihood covariance of the dynamics.
    - measurement_covariance: the maximum likelihood covariance of the measurements.
        """

    # get the batch shape and vector sizes

    batch_dim = tf.shape(measurement)[0]

    # make sure all inputs conform to the batch shape

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(measurement)[:-2],
        message="measurement should have scalar batch shape")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(mean)[:-2],
        message="mean should have same batch size as measurement")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(covariance)[:-2],
        message="covariance should have same batch size as measurement")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(controls)[:-2],
        message="controls should have same batch size as measurement")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(next_predicted_mean)[:-2],
        message="next predicted mean should have same batch size as measurement")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(next_predicted_covariance)[:-2],
        message="next predicted covariance should have same batch size as measurement")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(smooth_gain)[:-2],
        message="smooth gain should have same batch size as measurement")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(smooth_mean)[:-2],
        message="smooth mean should have same batch size as measurement")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(smooth_covariance)[:-2],
        message="smooth covariance should have same batch size as measurement")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(next_smooth_mean)[:-2],
        message="next smooth mean should have same batch size as measurement")

    tf.debugging.assert_equal(
        [batch_dim],
        tf.shape(next_smooth_covariance)[:-2],
        message="next smooth covariance should have same batch size as measurement")

    # compute the maximum likelihood parameters of the noise

    dynamics_error = next_smooth_mean - tf.matmul(
        dynamics_state_jacobian,
        smooth_mean) - tf.matmul(
            dynamics_controls_jacobian,
            controls)

    future_covariance = next_smooth_covariance - tf.matmul(
        tf.matmul(
            next_smooth_covariance,
            smooth_gain,
            transpose_b=True),
        dynamics_state_jacobian,
        transpose_b=True)

    future_covariance = future_covariance - tf.matmul(
        tf.matmul(
            dynamics_state_jacobian,
            smooth_gain),
        next_smooth_covariance)

    # compute the updated estimate of the system noise

    dynamics_covariance = future_covariance + tf.matmul(
        dynamics_error,
        dynamics_error,
        transpose_b=True)

    dynamics_covariance = dynamics_covariance + tf.matmul(
        dynamics_state_jacobian, tf.matmul(
            smooth_covariance,
            dynamics_state_jacobian,
            transpose_b=True))

    # compute the updated estimate of the measurement noise

    measurement_error = measurement - tf.matmul(
        measurement_jacobian, smooth_mean)

    # compute the updated estimate of the measurement noise

    measurement_covariance = tf.matmul(
        measurement_error,
        measurement_error,
        transpose_b=True)

    measurement_covariance = measurement_covariance + tf.matmul(
        measurement_jacobian, tf.matmul(
            smooth_covariance,
            measurement_jacobian,
            transpose_b=True))

    return (
        dynamics_covariance,
        measurement_covariance)


def kf_em_loop_body(
        measurements,
        mean,
        covariance,
        controls,
        predicted_mean,
        predicted_covariance,
        smooth_gain,
        smooth_mean,
        smooth_covariance,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        measurement_jacobian,
        dynamics_covariance_array,
        measurement_covariance_array,
        time
):
    """Inner body of kalman filter em update loop."""

    # run the kalman filter em step and collect results

    update_result = kf_em(
        measurements[:, time, :, :],
        mean[:, time, :, :],
        covariance[:, time, :, :],
        controls[:, time, :, :],
        predicted_mean[:, time + 1, :, :],
        predicted_covariance[:, time + 1, :, :],
        smooth_gain[:, time, :, :],
        smooth_mean[:, time, :, :],
        smooth_covariance[:, time, :, :],
        smooth_mean[:, time + 1, :, :],
        smooth_covariance[:, time + 1, :, :],
        dynamics_state_jacobian[:, time, :, :],
        dynamics_controls_jacobian[:, time, :, :],
        measurement_jacobian[:, time, :, :])

    # push the results into arrays for safe keeping

    dynamics_covariance_array = dynamics_covariance_array.write(
        time, update_result[0])

    measurement_covariance_array = measurement_covariance_array.write(
        time, update_result[1])

    time = time - 1

    return (
        measurements,
        mean,
        covariance,
        controls,
        predicted_mean,
        predicted_covariance,
        smooth_gain,
        smooth_mean,
        smooth_covariance,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        measurement_jacobian,
        dynamics_covariance_array,
        measurement_covariance_array,
        time)


def kf_em_loop_condition(
        measurements,
        mean,
        covariance,
        controls,
        predicted_mean,
        predicted_covariance,
        smooth_gain,
        smooth_mean,
        smooth_covariance,
        dynamics_state_jacobian,
        dynamics_controls_jacobian,
        measurement_jacobian,
        dynamics_covariance_array,
        measurement_covariance_array,
        time
):
    """Inner condition of kalman filter em update loop."""

    # exit the loop if time is less than zero

    return tf.greater(time, 0)
