# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

#
# STAIN: stain deconvolution from RGB image.
#

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__all__ = ['rgb2he', 'rgb2he_macenko']


from typing import Tuple
from scipy.linalg import eig
import numpy as np
from skimage.exposure import rescale_intensity


def rgb2he(img: np.ndarray, return_deconvolution_matrix: bool=False) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """Stain separation for H&E slides: estimate the H- and E- signal intensity
    and the residuals. Use Ruifrok's method:

    Ruifrok, Arnout C., and Dennis A. Johnston. "Quantification of
    histochemical staining by color deconvolution." Analytical and
    quantitative cytology and histology 23.4 (2001): 291-299.

    Args:
        img (numpy.ndarray): a H x W x 3 image array
        return_doconvolution_matrix: if True, the deconvolution matrix is also returned

    Returns:
        3 numpy arrays of size H x W with signal scaled to [0,1] corresponding
        to estimated intensities of Haematoxylin, Eosine and background/resodual
        components and the deconvolution matrix
    """
    # This implementation follows http://web.hku.hk/~ccsigma/color-deconv/color-deconv.html

    assert (img.ndim == 3)
    assert (img.shape[2] == 3)

    height, width, _ = img.shape

    img = -np.log((img + 1.0) / img.max())

    D = np.array([[ 1.92129515,  1.00941672, -2.34107612],
                  [-2.34500192,  0.47155124,  2.65616872],
                  [ 1.21495282, -0.99544467,  0.2459345 ]])

    rgb = img.swapaxes(2, 0).reshape((3, height*width))
    heb = np.dot(D, rgb)
    res_img = heb.reshape((3, width, height)).swapaxes(0, 2)

    return rescale_intensity(res_img[:,:,0], out_range=(0,1)), \
           rescale_intensity(res_img[:,:,1], out_range=(0,1)), \
           rescale_intensity(res_img[:,:,2], out_range=(0,1)), \
           D
## end rgb2he


def rgb2he_macenko(img, D=None, alpha=1.0, beta=0.15, white=255.0,
                   return_deconvolution_matrix=False) \
                   -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs stain separation from RGB images using the method in
    M Macenko, et al. "A method for normalizing histology slides for quantitative analysis",
    IEEE ISBI, 2009. dx.doi.org/10.1109/ISBI.2009.5193250

    Args:
        img (numpy.ndarray): RGB input image
        D (numpy.ndarray): a deconvolution matrix. If None, one will be computed from the image
        alpha (float): tolerance for pseudo-min/-max
        beta (float): OD threshold for transparent pixels
        white (float): white level (in each channel)
        return_deconvolution_matrix (bool): if True, the deconvolution matrix is also returned

    Returns:
        three 2d arrays for H-, E- and remainder channels, respectively.
        If return_deconvolution_matrix is True, the deconvolution matrix is also returned.
    """

    assert (img.ndim == 3)
    assert (img.shape[2] == 3)

    I = img.reshape((img.shape[0] * img.shape[1], 3))
    OD = -np.log((I + 1.0) / white)  # optical density

    if D is None:
        # the deconvolution matrix is not provided so one has to be estimated from the
        # image
        rows = (OD >= beta).all(axis=1)
        if not any(rows):
            # no rows with all pixels above the threshold
            raise RuntimeError('optical density below threshold')

        ODhat = OD[rows, :]  # discard transparent pixels

        u, V, _ = eig(np.cov(ODhat.T))
        idx = np.argsort(u)  # get a permutation to sort eigenvalues increasingly
        V = V[:, idx]        # sort eigenvectors
        theta = np.dot(ODhat, V[:, 1:3])  # project optical density onto the eigenvectors
                                          # corresponding to the largest eigenvalues
        phi = np.arctan2(theta[:,1], theta[:,0])
        min_phi, max_phi = np.percentile(phi, [alpha, 100.0-alpha], axis=None)

        u1 = np.dot(V[:,1:3], np.array([[np.cos(min_phi)],[np.sin(min_phi)]]))
        u2 = np.dot(V[:,1:3], np.array([[np.cos(max_phi)],[np.sin(max_phi)]]))

        if u1[0] > u2[0]:
            D = np.hstack((u1, u2)).T
        else:
            D = np.hstack((u2, u1)).T

        D = np.vstack((D, np.cross(D[0,],D[1,])))
        D = D / np.reshape(np.repeat(np.linalg.norm(D, axis=1), 3), (3,3), order=str('C'))

    img_res = np.linalg.solve(D.T, OD.T).T
    img_res = np.reshape(img_res, img.shape, order=str('C'))

    if not return_deconvolution_matrix:
        D = None

    return rescale_intensity(img_res[:,:,0], out_range=(0,1)), \
           rescale_intensity(img_res[:,:,1], out_range=(0,1)), \
           rescale_intensity(img_res[:,:,2], out_range=(0,1)), \
           D
# end rgb2he_macenko
