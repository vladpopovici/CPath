#   -*- coding: utf-8 -*-
#
#  --------------------------------------------------------------------
#  Copyright (c) 2022 Vlad Popovici <popovici@bioxlab.org>
#
#  Licensed under the MIT License. See LICENSE file in root folder.
#  --------------------------------------------------------------------

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = 0.2
__all__ = ['MRI']

import shapely.geometry as shg
import shapely.affinity as sha
import zarr
import pathlib
import numpy as np
import tifffile
from typing import Tuple

from .wsi import WSIInfo


#####
class MRI(object):
    """MultiResolution Image - a simple and convenient interface to access pixels from a
    pyramidal image (e.g. in OME TIFF format). The image is accessed as a ZARR store and the
    details for image and pyramid layout are passed by the WSIInfo object.

    Args:
        wsi (WSIInfo): an object containing the meta-data about the image, including its
            path. See cpath.core.WSIInfo for details.

    Attributes:
        _wsi (WSIInfo): meta-data about the image
        _pyramid_levels (array): a 2 x <n_levels> array with pyramid layer extents. It is 
            a shortcut for the corresponding info in WSIInfo
        _downsample_factors (array): a 1 x <n_levels> array with scaling factors for each
            level in the pyramid
    """

    def __init__(self, wsi: WSIInfo):
        """Initialize a multi-resolution image (MRI) with info from associated WSIInfo."""

        self._wsi = wsi

        n_levels = wsi.level_count()
        self._pyramid_levels = np.zeros((2, n_levels), dtype=int)
        self._downsample_factors = np.zeros((n_levels,), dtype=int)
        for lv in range(0, n_levels):
            self._pyramid_levels[0, lv] = self._wsi._pyramid_levels[lv]['max_x']
            self._pyramid_levels[1, lv] = self._wsi._pyramid_levels[lv]['max_y']
            self._downsample_factors[lv] = self._wsi.magnif_converter._magnif_step ** lv
        return
    
    def _make_index(self, ix: slice, iy: slice, ic: slice) -> Tuple[int]:
        idx = [0,0,0]
        idx[self._wsi.ix()] = ix
        idx[self._wsi.iy()] = iy
        idx[self._wsi.ic()] = ic
        
        return tuple(idx)
    
    @property
    def path(self) -> pathlib.Path:
        return self._wsi.path

    @property
    def widths(self) -> np.array:
        # All widths for the pyramid levels
        return self._pyramid_levels[0, :]

    @property
    def heights(self) -> np.array:
        # All heights for the pyramid levels
        return self._pyramid_levels[1, :]

    def extent(self, level: int = 0) -> Tuple[int]:
        # width, height for a given level
        return tuple(self._pyramid_levels[:, level])

    def level_shape(self, level: int=0) -> dict[str, int]:
        return {'width': self._pyramid_levels[0, level],
                'height': self._pyramid_levels[1, level]}

    @property
    def nlevels(self) -> int:
        return self._pyramid_levels.shape[1]

    def between_level_scaling_factor(self, from_level: int, to_level: int) -> float:
        """Return the scaling factor for converting coordinates (magnification)
        between two levels in the MRI.

        Args:
            from_level (int): original level
            to_level (int): destination level

        Returns:
            float
        """
        f = self._downsample_factors[from_level] / self._downsample_factors[to_level]

        return f

    def convert_px(self, point, from_level, to_level):
        """Convert pixel coordinates of a point from <from_level> to
        <to_level>

        Args:
            point (tuple): (x,y) coordinates in <from_level> plane
            from_level (int): original image level
            to_level (int): destination level

        Returns:
            x, y (float): new coodinates - no rounding is applied
        """
        if from_level == to_level:
            return point  # no conversion is necessary
        x, y = point
        f = self.between_level_scaling_factor(from_level, to_level)
        x *= f
        y *= f

        return x, y

    def get_region_px(self, x0: int, y0: int,
                      width: int, height: int,
                      level: int = 0, as_type=np.uint8) -> np.ndarray:
        """Read a region from the image source. The region is specified in
            pixel coordinates.

            Args:
                x0, y0 (long): top left corner of the region (in pixels, at the specified
                level)
                width, height (long): width and height (in pixels) of the region.
                level (int): the magnification level to read from
                as_type: type of the pixels (default numpy.uint8)

            Returns:
                a numpy.ndarray [height x width x channels]
        """

        if level < 0 or level >= self.nlevels:
            raise RuntimeError("requested level does not exist")

        # check bounds:
        if x0 >= self.widths[level] or y0 >= self.heights[level] or \
                x0 + width > self.widths[level] or \
                y0 + height > self.heights[level]:
            raise RuntimeError("region out of layer's extent")

        idx = self._make_index(slice(x0, x0+width), slice(y0, y0+height), slice(0,3))

        tif = tifffile.imread(self._wsi.path, aszarr=True)
        with zarr.open(tif, mode='r') as z:
            lv = self._wsi._pyramid_levels[level]['path']
            img = np.array(z[lv][idx], dtype=as_type)
        img = np.moveaxis(
            img, 
            [self._wsi.iy(), self._wsi.ix(), self._wsi.ic()], 
            [0, 1, 2]
            )

        return img


    def get_plane(self, level: int = 0, as_type=np.uint8) -> np.ndarray:
        """Read a whole plane from the image pyramid and return it as a Numpy array.

        Args:
            level (int): pyramid level to read
            as_type: type of the pixels (default numpy.uint8)

        Returns:
            a numpy.ndarray
        """
        if level < 0 or level >= self.nlevels:
            raise RuntimeError("requested level does not exist")
        
        return self.get_region_px(0, 0, self.widths[level], self.heights[level], level, as_type)


    def get_polygonal_region_px(self, contour: shg.Polygon, level: int,
                                border: int = 0, as_type=np.uint8) -> np.ndarray:
        """Returns a rectangular view of the image source that minimally covers a closed
        contour (polygon). All pixels outside the contour are set to 0.

        Args:
            contour (shapely.geometry.Polygon): a closed polygonal line given in
                terms of its vertices. The contour's coordinates are supposed to be
                precomputed and to be represented in pixel units at the desired level.
            level (int): image pyramid level
            border (int): if > 0, take this many extra pixels in the rectangular
                region (up to the limits on the image size)
            as_type: pixel type for the returned image (array)

        Returns:
            a numpy.ndarray
        """
        x0, y0, x1, y1 = [int(_z) for _z in contour.bounds]
        x0, y0 = max(0, x0 - border), max(0, y0 - border)
        x1, y1 = min(x1 + border, self.extent(level)[0]), \
                 min(y1 + border, self.extent(level)[1])
        # Shift the annotation such that (0,0) will correspond to (x0, y0)
        contour = sha.translate(contour, -x0, -y0)

        # Read the corresponding region
        img = self.get_region_px(x0, y0, x1 - x0, y1 - y0, level, as_type=np.uint8)

        # mask out the points outside the contour
        for i in np.arange(img.shape[0]):
            # line mask
            lm = np.zeros((img.shape[1],), dtype=img.dtype)
            j = [_j for _j in np.arange(img.shape[1]) if shg.Point(_j, i).within(contour)]
            lm[j] = 1
            img[i,] = img[i,] * lm

        return img

##
