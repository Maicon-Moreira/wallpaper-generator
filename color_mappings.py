import numba as nb
import math


@nb.njit()
def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return [v, v, v]
    i = int(h * 6.0)  # XXX assume int() truncates!
    f = (h * 6.0) - i
    p, q, t = v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f))
    i %= 6
    if i == 0:
        return [v, t, p]
    if i == 1:
        return [q, v, p]
    if i == 2:
        return [p, v, t]
    if i == 3:
        return [p, q, v]
    if i == 4:
        return [t, p, v]
    if i == 5:
        return [v, p, q]


# THE FOLLOWING FUNCTIONS ARE USED FOR HCL COLOR SPACE MAPPING
# 
# HCL (Hue-Chroma-Luminance) or LCh refers to any of the many cylindrical
# color space models that are designed to accord with human
# perception of color with the three parameters.
# 
# REFERENCES:
# https://en.wikipedia.org/wiki/HCL_color_space
# https://stackoverflow.com/questions/7530627/hcl-color-to-rgb-and-backward
# https://gist.github.com/pushkine/c8ba98294233d32ab71b7e19a0ebdbb9


@nb.njit()
def rgb255(v):
    return min(max(v, 0), 255)


@nb.njit()
def b1(v):
    if v > 0.0031308:
        return v ** (1 / 2.4) * 269.025 - 14.025
    else:
        return v * 3294.6


@nb.njit()
def b2(v):
    if v > 0.2068965:
        return v**3
    else:
        return (v - 4 / 29) * (108 / 841)


@nb.njit()
def a1(v):
    if v > 10.314724:
        return ((v + 14.025) / 269.025) ** 2.4
    else:
        return v / 3294.6


@nb.njit()
def a2(v):
    if v > 0.0088564:
        return v ** (1 / 3)
    else:
        return v / (108 / 841) + 4 / 29


@nb.njit()
def hcl_to_rgb(h, c, l):
    y = b2((l + 16) / 116)
    x = b2((l + 16) / 116 + (c / 500) * math.cos(h * math.pi / 180))
    z = b2((l + 16) / 116 - (c / 200) * math.sin(h * math.pi / 180))
    return (
        rgb255(b1(x * 3.021973625 - y * 1.617392459 - z * 0.404875592)),
        rgb255(b1(x * -0.943766287 + y * 1.916279586 + z * 0.027607165)),
        rgb255(b1(x * 0.069407491 - y * 0.22898585 + z * 1.159737864)),
    )
