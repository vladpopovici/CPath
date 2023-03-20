#   -*- coding: utf-8 -*-
#
#  --------------------------------------------------------------------
#  Copyright (c) 2022-2023 Vlad Popovici <popovici@bioxlab.org>
#
#  Licensed under the MIT License. See LICENSE file in root folder.
#  --------------------------------------------------------------------
__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.1

__all__ = ['NumpyImage', 'R_', 'G_', 'B_', 'mark_points', 'array_to_image']

import numpy as np
from typing import NewType
import matplotlib.pyplot
import skimage.draw as skd


ImageShape = NewType("ImageShape", dict[str, int])

#####
class NumpyImage:
    """This is merely a namespace for collecting a number of useful
    functions that are applied to images stored as Numpy arrays.
    Usually, such an image -either single channel or 3(4) channels -
    is stored as a H x W (x C) array, with H (height) rows and W (width)
    columns. C=3 or 4.
    """

    @staticmethod
    def width(img: np.ndarray) -> int:
        return img.shape[1]

    @staticmethod
    def height(img: np.ndarray) -> int:
        return img.shape[0]

    @staticmethod
    def nchannels(img: np.ndarray) -> int:
        if img.ndim > 2:
            return img.shape[2]
        else:
            return 1

    @staticmethod
    def is_empty(img: np.array, empty_level: float=0) -> bool:
        """Is the image empty?

        Args:
            img (numpy.ndarray): image
            empty_level (int/numeric): if the sum of pixels is at most this
                value, the image is considered empty.

        Returns:
            bool
        """

        return img.sum() <= empty_level

    @staticmethod
    def is_almost_white(img: np.array, almost_white_level: float=254, max_stddev: float=1.5) -> bool:
        """Is the image almost white?

        Args:
            img (numpy.ndarray): image
            almost_white_level (int/numeric): if the average intensity per channel
                is above the given level, decide "almost white" image.
            max_stddev (float): max standard deviation for considering the image
                almost constant.

        Returns:
            bool
        """

        return (img.mean() >= almost_white_level) and (img.std() <= max_stddev)


def R_(_img: np.ndarray) -> np.ndarray:
    return _img[:, :, 0]


def G_(_img: np.ndarray) -> np.ndarray:
    return _img[:, :, 1]


def B_(_img: np.ndarray) -> np.ndarray:
    return _img[:, :, 2]


def array_to_image(filename, X, cmap=matplotlib.cm.plasma, dpi=120.0,
                   invert_y=True):
    """Produce a visual representation of a data matrix.

    Parameters:
        :param filename: str
            name of the file to save the image to
        :param X: numpy.array (2D)
            the data matrix to be converted to a raster image
        :param cmap:  matplotlib.cm
            color map
        :param dpi: float
            image resolution (DPI)
        :param invert_y: bool
            should the y-axis (rows) be inverted, such that the top
            of the matrix (low row counts) would correspond to low
            y-values?
    """

    # From SciPy cookbooks, https://scipy-cookbook.readthedocs.io/items/Matplotlib_converting_a_matrix_to_a_raster_image.html

    figsize = (np.array(X.shape) / float(dpi))[::-1]
    matplotlib.rcParams.update({'figure.figsize': figsize})
    fig = matplotlib.pyplot.figure(figsize=figsize)
    matplotlib.pyplot.axes([0, 0, 1, 1])  # Make the plot occupy the whole canvas
    matplotlib.pyplot.axis('off')
    fig.set_size_inches(figsize)

    matplotlib.pyplot.imshow(X, origin='upper' if invert_y else 'lower',
                             aspect='equal', cmap=cmap)

    matplotlib.pyplot.savefig(filename, facecolor='white', edgecolor='black', dpi=dpi)
    matplotlib.pyplot.close(fig)

    return


def mark_points(image: np.array, points: np.array, radius: int, color: tuple,
                in_situ: bool = True) -> np.array:
    """Mark a series of points in an image.

    Parameters:
        :param image: an array in which to mark the points. May be multi-channel.
        :param points: a N X 2 array with (row, col) coords for the N points
        :param radius: the radius of the disc to be drawn at the positions
        :param color: a tuple with R,G,B color specifications for the marks. If the
            image is single channel, only the first value will be used (R)
        :param in_situ: (bool) if True, the points are marked directly in the image,
            otherwise a copy will be used

    Return:
        an array with the points marked in the image
    """

    if in_situ:
        res = image
    else:
        res = image.copy()

    if res.ndim == 2:
        col = color[0]
    else:
        col = color[:3]

    for p in points:
        r, c = skd.circle(p[0], p[1], radius, shape=res.shape)
        res[r,c,...] = col

    return res
    