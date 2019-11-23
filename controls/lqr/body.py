"""Author: Brandon Trabucco, Copyright 2019, MIT License"""


from controls.lqr.update import lqr_update


def lqr_body(
        Vxx,
        Vx,
        Fx,
        Fu,
        Cxx,
        Cxu,
        Cux,
        Cuu,
        Cx,
        Cu,
        Qxx_array,
        Qxu_array,
        Qux_array,
        Quu_array,
        Qx_array,
        Qu_array,
        Kx_array,
        k_array,
        Vxx_array,
        Vx_array,
        time,
        horizon
):
    Qxx, Qxu, Qux, Quu, Qx, Qu, Kx, k, Vxx, Vx = lqr_update(
            Vxx,
            Vx,
            Fx[time, :, :, :],
            Fu[time, :, :, :],
            Cxx[time, :, :, :],
            Cxu[time, :, :, :],
            Cux[time, :, :, :],
            Cuu[time, :, :, :],
            Cx[time, :, :, :],
            Cu[time, :, :, :])
    return (
        Vxx,
        Vx,
        Fx,
        Fu,
        Cxx,
        Cxu,
        Cux,
        Cuu,
        Cx,
        Cu,
        Qxx_array.write(time, Qxx),
        Qxu_array.write(time, Qxu),
        Qux_array.write(time, Qux),
        Quu_array.write(time, Quu),
        Qx_array.write(time, Qx),
        Qu_array.write(time, Qu),
        Kx_array.write(time, Kx),
        k_array.write(time, k),
        Vxx_array.write(time, Vxx),
        Vx_array.write(time, Vx),
        time - 1,
        horizon)
