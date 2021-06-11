import numpy as np
import matplotlib.pyplot as plt


def get_frequencies(img: np.ndarray, epsilon: float):
    """
    Apply fft, compute magnitude of signal and azimuthal average

    :param img: input image that we calculate from
    :param epsilon: we add this value to the frequency such that we don't get numerical errors while applying log
    :return: the resulting one dimensional value
    """
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    fshift += epsilon

    # magnitude_spectrum = np.abs(fshift)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    psd1D = azimuthal_average(magnitude_spectrum)
    # psd1D = azimuthal_average_v2(magnitude_spectrum)

    return psd1D


def azimuthal_average(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    :param image - The 2D image
    :param center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).
    """
    # Calculate the indices from the image (coordinates of each pixel)
    y, x = np.indices(image.shape)

    # Center the image: shift each coordinate by (x_center,y_center) such that the center pixel has coordinates (0,0)
    # TODO: Shouldn't we use // ?
    if not center:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    # Compute the distance from each pixel to the center
    r = np.hypot(x - center[0], y - center[1])
    # r = np.maximum((x-center[0])**2, (y-center[1])**2)

    # Get sorted radii in a flattened array ind: order of indices such that r.flat[ind](r_sorted) is a sorted version
    # of r i_sorted: sort the corresponding pixel values we get groups of pixel values grouped together,
    # because they are all equally far from the center => they are on the same radius
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin. Where the radius value changes, we get a 1 (for example if N
    # = 3 and N+1 = 4 => get 1 at position 4, but if N = 3 and N+1 = 3 get 0, there is no change)
    # Then, select all values that are 1
    # After, compute how many radii have a specific value
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof


def azimuthal_average_v2(image):
    # create array of radii
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    y_center = image.shape[0] // 2
    x_center = image.shape[1] // 2
    x = x - x_center
    y = y - y_center

    R = np.sqrt(x ** 2 + y ** 2)

    # calculate the mean
    f = lambda r: image[(R >= r - 0.5) & (R < r + 0.5)].mean()
    max_val = min(image.shape[0] // 2, image.shape[1] // 2)
    r = np.linspace(1, max_val, num=max_val)
    mean = np.vectorize(f)(r)

    return mean


def split_image(img, fraction: int = 1):
    """
    Divides image in fraction x fraction equally-sized blocks
    :param img: Image to be divided
    :param fraction: Each block will have size ((img_width)/fraction, (img_height)/fraction) => there will be fraction x fraction
    blocks in total
    :return: ndarray with the stacked images
    """
    # pieces = np.power(fraction, 2)

    new_height = img.shape[0] // fraction
    new_width = img.shape[1] // fraction

    img_resized = img[:fraction * new_height, :fraction * new_width]
    shape_h = img_resized.shape[0] - new_height
    shape_w = img_resized.shape[1] - new_width

    blocks = [img[h:h + new_height, w:w + new_width] for h in range(0, shape_h + 1, new_height)
              for w in range(0, shape_w + 1, new_width)]
    # import matplotlib.pyplot as plt; plt.imshow(blocks[0]); plt.show()
    return blocks
