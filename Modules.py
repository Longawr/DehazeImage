import cv2
import numpy as np
from numpy.linalg import inv
from collections import defaultdict
from itertools import combinations_with_replacement

R, G, B = 0, 1, 2  # index for convenience

def get_illumination_channel(I, w):
    """
    Calculate dark and bright channels of the image.

    Parameters:
    - I: Input image.
    - w: Window size for the calculation.

    Returns:
    - Dark channel image.
    - Bright channel image.
    """
    M, N, _ = I.shape
    darkch = np.zeros((M, N))
    brightch = np.zeros((M, N))

    padded = np.pad(I, ((w//2, w//2), (w//2, w//2), (0, 0)), 'edge')

    for i, j in np.ndindex(darkch.shape):
        darkch[i, j] = np.min(padded[i:i + w, j:j + w, :]) # dark channel
        brightch[i, j] = np.max(padded[i:i + w, j:j + w, :]) # bright channel

    return darkch, brightch

def get_atmosphere(I, brightch, p=0.1):
    """
    Estimate the global atmospheric light from the brightest pixels in the bright channel.

    Parameters:
    - I: Input image.
    - brightch: Bright channel image.
    - p: Percentage of the brightest pixels to consider.

    Returns:
    - Estimated atmospheric light.
    """
    M, N = brightch.shape
    flatI = I.reshape(M*N, 3)  # Reshaping the image array
    flatbright = brightch.ravel()  # Flattening the image array

    searchidx = (-flatbright).argsort()[:int(M*N*p)]  # Sorting and slicing
    A = np.mean(flatI.take(searchidx, axis=0), dtype=np.float64, axis=0)
    return A

def get_initial_transmission(A, brightch):
    """
    Calculate the initial transmission map that is min-max normalized from the global bright and bright channel.

    Parameters:
    - A: Estimated atmospheric light.
    - brightch: Bright channel image.

    Returns:
    - Normalized initial transmission map.
    """
    A_c = np.max(A)
    init_t = (brightch-A_c)/(1.-A_c) # finding initial transmission map
    return (init_t - np.min(init_t))/(np.max(init_t) - np.min(init_t)) # normalized initial transmission map

def reduce_init_t(init_t):
    """
    Reduce the initial transmission map by adjusting the values using a lookup table.

    Parameters:
    - init_t: Normalized initial transmission map.

    Returns:
    - Reduced normalized initial transmission map.
    """
    init_t = (init_t*255).astype(np.uint8) 
    xp = [0, 32, 255]
    fp = [0, 32, 48]
    x = np.arange(256) # creating array [0,...,255]
    table = np.interp(x, xp, fp).astype('uint8') # interpreting fp according to xp in range of x
    init_t = cv2.LUT(init_t, table) # lookup table
    init_t = init_t.astype(np.float64)/255 # normalizing the transmission map
    return init_t

def get_corrected_transmission(I, A, darkch, brightch, init_t, alpha, omega, w):
    """
    Calculate the corrected transmission map based on the initial transmission map and other parameters.

    Parameters:
    - I: Input image.
    - A: Estimated atmospheric light.
    - darkch: Dark channel image.
    - brightch: Bright channel image.
    - init_t: Initial transmission map.
    - alpha: Adjustment parameter.
    - omega: Adjustment parameter.
    - w: Window size for calculation.

    Returns:
    - Corrected transmission map.
    """
    im = np.empty(I.shape, I.dtype)
    for ind in range(0, 3):
        im[:, :, ind] = I[:, :, ind] / A[ind]  # Divide pixel values by atmospheric light
    dark_c, _ = get_illumination_channel(im, w)  # Dark channel transmission map
    dark_t = 1 - omega * dark_c  # Corrected dark transmission map
    corrected_t = init_t  # Initializing corrected transmission map with initial transmission map
    diffch = brightch - darkch  # Difference between transmission maps

    for i in range(diffch.shape[0]):
        for j in range(diffch.shape[1]):
            if diffch[i, j] < alpha:
                corrected_t[i, j] = dark_t[i, j] * init_t[i, j]

    return np.abs(corrected_t)

def boxfilter(I, r):
    """
    Apply box filtering to the image.

    Parameters:
    - I: Input image.
    - r: Size of the filter.

    Returns:
    - Filtered 2D array representing.
    """
    M, N = I.shape
    dest = np.zeros((M, N))
    
    # cumulative sum over Y axis (tate-houkou no wa)
    sumY = np.cumsum(I, axis=0)
    
    # difference over Y axis
    dest[:r + 1] = sumY[r:2*r + 1] # top r+1 lines
    dest[r + 1:M - r] = sumY[2*r + 1:] - sumY[:M - 2*r - 1]
    
    #tile replicate sumY[-1] and line them up to match the shape of (r, 1)
    dest[-r:] = np.tile(sumY[-1], (r, 1)) - sumY[M - 2*r - 1:M - r - 1] # bottom r lines

    # cumulative sum over X axis
    sumX = np.cumsum(dest, axis=1)
    
    # difference over X axis
    dest[:, :r + 1] = sumX[:, r:2*r + 1] # left r+1 columns
    dest[:, r + 1:N - r] = sumX[:, 2*r + 1:] - sumX[:, :N - 2*r - 1]
    dest[:, -r:] = np.tile(sumX[:, -1][:, None], (1, r)) - sumX[:, N - 2*r - 1:N - r - 1] # right r columns

    return dest

def guided_filter(I, p, r=10, eps=1e-3):
    """
    Apply guided filter to the input image.
    
    Parameters:
        - I: Guidance image.
        - p: Input image.
        - r: Radius of the filter window.
        - eps: Regularization parameter.
    
    Returns:
        - Filtered image.
    """
    M, N = p.shape
    base = boxfilter(np.ones((M, N)), r)
    
    means = [boxfilter(I[:, :, i], r) / base for i in range(3)]
    mean_p = boxfilter(p, r) / base
    means_IP = [boxfilter(I[:, :, i] * p, r) / base for i in range(3)]
    
    covIP = [means_IP[i] - means[i] * mean_p for i in range(3)]
    var = defaultdict(dict)
    
    for i, j in combinations_with_replacement(range(3), 2):
        var[i][j] = boxfilter(I[:, :, i] * I[:, :, j], r) / base - means[i] * means[j]

    a = np.zeros((M, N, 3))
    for y, x in np.ndindex(M, N):
        Sigma = np.array([[var[R][R][y, x], var[R][G][y, x], var[R][B][y, x]],
                          [var[R][G][y, x], var[G][G][y, x], var[G][B][y, x]],
                          [var[R][B][y, x], var[G][B][y, x], var[B][B][y, x]]])
        cov = np.array([c[y, x] for c in covIP])
        a[y, x] = np.dot(cov, inv(Sigma + eps * np.eye(3)))

    b = mean_p - a[:, :, R] * means[R] - a[:, :, G] * means[G] - a[:, :, B] * means[B]
    q = (boxfilter(a[:, :, R], r) * I[:, :, R] + boxfilter(a[:, :, G], r) * I[:, :, G] +
         boxfilter(a[:, :, B], r) * I[:, :, B] + boxfilter(b, r)) / base

    return q

def get_final_image(I, A, refined_t, tmin):
    """
    Compute the final enhanced image using the estimated atmospheric light and transmission map.
    
    Parameters:
        - I: Input image.
        - A: Estimated atmospheric light.
        - refined_t: Refined transmission map.
        - tmin: Minimum threshold for transmission.
        
    Returns:
        - Final enhanced image.
    """
    refined_t_broadcasted = np.broadcast_to(refined_t[:, :, None], (refined_t.shape[0], refined_t.shape[1], 3))  # Duplicate the channel of 2D refined map to 3 channels
    J = (I - A) / (np.where(refined_t_broadcasted < tmin, tmin, refined_t_broadcasted)) + A  # Compute the result 
    
    return (J - np.min(J)) / (np.max(J) - np.min(J))  # Normalize the image