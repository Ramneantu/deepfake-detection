# from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
import numpy as np

# Authors of this file: Ricard Durall, Janis Keuper
# Edited in the context of the EAI Praktikum SoSe2021

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
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

    # Get sorted radii in a flattened array ind: order of indeces such that r.flat[ind](r_sorted) is a sorted version
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
