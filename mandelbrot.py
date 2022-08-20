from PIL import Image
import numba as nb
import numpy as np
from tqdm import tqdm
import time


NJIT_MODE = True


@nb.jit(nopython=NJIT_MODE)
def mandelbrot_iterations(real, imag, max_iterations):
    """Determine the number of iterations for a complex number to escape the Mandelbrot set."""
    c = complex(real, imag)
    z = 0
    for i in range(max_iterations):
        z = z * z + c
        if abs(z) > 2:
            return i
    return i


@nb.jit(nopython=NJIT_MODE)
def generate_mandelbrot_iterations_array(x1, y1, x2, y2, width, height, max_iterations):
    """Generate the Mandelbrot iterations array."""
    x_step = (x2 - x1) / width
    y_step = (y2 - y1) / height
    iterations = np.zeros((width, height), dtype=np.uint16)
    for x in range(width):
        for y in range(height):
            iterations[x, y] = mandelbrot_iterations(
                x1 + x * x_step, y1 + y * y_step, max_iterations
            )
    return iterations


@nb.jit(nopython=NJIT_MODE)
def map_mandelbrot_iterations_to_grayscale(iterations, max_iterations):
    """Map the Mandelbrot iterations array to a grayscale image."""
    image = np.zeros((iterations.shape[0], iterations.shape[1]), dtype=np.uint8)
    for x in range(iterations.shape[0]):
        for y in range(iterations.shape[1]):
            if iterations[x, y] == max_iterations:
                image[x, y] = 0
            else:
                image[x, y] = 255 - int(iterations[x, y] * 255 / max_iterations)
    return image


def average(*args):
    return sum(args) / len(args)


def render_mandelbrot(
    center_x, center_y, zoom, pixels_width, pixels_height, max_iterations, filename
):
    scaled_zoom = zoom * average(pixels_width, pixels_height)
    width, height = pixels_width / scaled_zoom, pixels_height / scaled_zoom
    x1, x2 = center_x - width / 2, center_x + width / 2
    y1, y2 = center_y - height / 2, center_y + height / 2
    print(f"Rendering Mandelbrot set with center ({center_x}, {center_y}) and zoom {zoom}")
    print(f"x1 = {x1}, x2 = {x2}, y1 = {y1}, y2 = {y2}")

    print('-'*32)
    print("Generating Mandelbrot iterations array...", end="")
    start = time.time()
    iterations = generate_mandelbrot_iterations_array(
        x1, y1, x2, y2, pixels_width, pixels_height, max_iterations
    )
    print(f" done in {time.time() - start:.2f} seconds")

    print('Mapping Mandelbrot iterations to grayscale...', end="")
    start = time.time()
    image = Image.fromarray(
        map_mandelbrot_iterations_to_grayscale(iterations, max_iterations).T
    )
    print(f" done in {time.time() - start:.2f} seconds")

    print("Saving image...", end="")
    start = time.time()
    image.save("mandelbrot.png")
    print(f" done in {time.time() - start:.2f} seconds")


def main():
    render_mandelbrot(-0.5, 0, 1, 1920, 1080, 100, "mandelbrot.png")


if __name__ == "__main__":
    main()
