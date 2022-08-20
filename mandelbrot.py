from PIL import Image
import numba as nb
import numpy as np
from tqdm import tqdm
import time


@nb.njit()
def mandelbrot_iterations(real, imag, max_iterations):
    """Determine the number of iterations for a complex number to escape the Mandelbrot set."""
    c = complex(real, imag)
    z = 0
    for i in range(max_iterations):
        z = z * z + c
        if abs(z) > 2:
            return i
    return i


@nb.njit(parallel=True)
def generate_mandelbrot_iterations_array(x1, y1, x2, y2, width, height, max_iterations):
    """Generate the Mandelbrot iterations array."""
    x_step = (x2 - x1) / width
    y_step = (y2 - y1) / height
    iterations = np.zeros((width, height), dtype=np.uint16)
    for x in nb.prange(width):
        for y in range(height):
            iterations[x, y] = mandelbrot_iterations(
                x1 + x * x_step, y1 + y * y_step, max_iterations
            )
    return iterations


@nb.njit(parallel=True)
def map_mandelbrot_iterations_to_grayscale(iterations, max_iterations):
    """Map the Mandelbrot iterations array to a grayscale image."""
    image = np.zeros((iterations.shape[0], iterations.shape[1], 3), dtype=np.uint8)
    for x in nb.prange(iterations.shape[0]):
        for y in range(iterations.shape[1]):
            if iterations[x, y] == max_iterations:
                image[x, y] = 0
            else:
                color = int(255 * iterations[x, y] / max_iterations)
                image[x, y] = color, color, color
    return image


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


@nb.njit()
def map_mandelbrot_iterations_to_hsv(iterations, max_iterations):
    """Map the Mandelbrot iterations array to a grayscale image."""
    image = np.zeros((iterations.shape[0], iterations.shape[1], 3), dtype=np.uint8)
    for x in nb.prange(iterations.shape[0]):
        for y in range(iterations.shape[1]):
            if iterations[x, y] == max_iterations - 1:
                image[x, y] = 0
            else:
                hsv = [
                    np.power((iterations[x, y] / max_iterations) * 360, 1.25) % 360,
                    100,
                    100,
                ]
                rgb = hsv_to_rgb(hsv[0] / 360, hsv[1] / 100, hsv[2] / 100)
                image[x, y] = [int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)]
    return image


def average(*args):
    return sum(args) / len(args)


color_mapping_map = {
    "grayscale": map_mandelbrot_iterations_to_grayscale,
    "hsv": map_mandelbrot_iterations_to_hsv,
}


def render_mandelbrot(
    center_x,
    center_y,
    zoom,
    resolution,
    max_iterations,
    filename,
    color_mapping="grayscale",
):
    pixels_width, pixels_height = resolution
    scaled_zoom = zoom * average(pixels_width, pixels_height)
    width, height = pixels_width / scaled_zoom, pixels_height / scaled_zoom
    x1, x2 = center_x - width / 2, center_x + width / 2
    y1, y2 = center_y - height / 2, center_y + height / 2
    print(
        f"Rendering Mandelbrot set with center ({center_x}, {center_y}) and zoom {zoom}"
    )
    print(f"x1 = {x1}, x2 = {x2}, y1 = {y1}, y2 = {y2}")

    print("-" * 32)
    print("Generating Mandelbrot iterations array...", end="")
    start = time.time()
    iterations = generate_mandelbrot_iterations_array(
        x1, y1, x2, y2, pixels_width, pixels_height, max_iterations
    )
    print(f" done in {time.time() - start:.2f} seconds")

    print("Mapping Mandelbrot iterations to grayscale...", end="")
    start = time.time()
    image = Image.fromarray(
        color_mapping_map[color_mapping](iterations, max_iterations).transpose(1, 0, 2)
    )
    print(f" done in {time.time() - start:.2f} seconds")

    print("Saving image...", end="")
    start = time.time()
    image.save("mandelbrot.png")
    print(f" done in {time.time() - start:.2f} seconds")


def main():

    HD = (1280, 720)
    FHD = (1920, 1080)
    QHD = (2560, 1440)
    UHD = (3840, 2160)
    FUHD = (7680, 4320)
    UW_FHD = (2560, 1080)
    UW_QHD = (3440, 1440)
    UW_UHD = (5120, 2160)
    UW_FUHD = (10240, 4320)

    render_mandelbrot(-0.74, -0.15, 75, HD, 500, "mandelbrot.png", "hsv")


if __name__ == "__main__":
    main()
