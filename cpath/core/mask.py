#   -*- coding: utf-8 -*-
#
#  --------------------------------------------------------------------
#  Copyright (c) 2022 Vlad Popovici <popovici@bioxlab.org>
#
#  Licensed under the MIT License. See LICENSE file in root folder.
#  --------------------------------------------------------------------

# MASKS are single channel images (arrays) storing information about a
# corresponding pixel in the image. A binary mask, for example, indicates
# that the pixels corresponding to "True" values in the mask have a common
# property (e.g. they belong to the same object).
#
# Usage philosophy: a mask is created to match the image shape (width and
# height) of the image currently processed. This mask is stored as a temporary
# ZARR array, in <tmp_folder>. Upon completion of the processing, the user must
# request that the mask is written to its final destination: either a large
# TIFF file or a ZARR pyramid. The temporary file is automatically deleted
# when the mask is closed (mask object expires).

import numpy as np
from pathlib import Path
import shutil
import zarr
from hashlib import md5
from time import localtime
import tifffile as tif
from skimage.transform import resize


# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

__author__ = "Vlad Popovici <popovici@bioxlab.org>"

#
# CPATH.MASK - various functions for creating and manipulating image
# masks (i.e. binary images of 0s and 1s).
#

__all__ = [
    'binary_mask', 
    'mask_to_external_contours', 
    'add_region', 
    'masked_points', 
    'apply_mask',
    'BinaryMask'
    ]

import numpy as np
import skimage.draw
import cv2
import shapely.geometry as shg
from .annot import Polygon

##-
def binary_mask(image: np.ndarray, level: float, mode: str = 'exact') -> np.ndarray:
    """Convert a single channel image into a binary mask by simple
    thresholding. This is a convenience function, smarter ways for
    binarizing images exist.

    Args:
        image (numpy.array): a 2-dimensional array
        level (float): level for binarization
        mode (str): binarization strategy:
            'exact': pixels having value 'level' are set to 1, all others to 0
            'above': pixels having value > 'level' are set 1, all others to 0
            'below': pixels having value < 'level' are set 1, all others to 0

    Returns:
        a numpy.array of type 'uint8' and same shape as <image>
    """
    mode = mode.lower()
    if mode not in ['exact', 'above', 'below']:
        raise RuntimeError('unknown mode: ' + mode)

    if image.ndim != 2:
        raise RuntimeError('<image> must be single channel!')

    level = np.cast[image.dtype](level).item()  # need to convert to image dtype for == to work well

    mask = np.zeros_like(image, dtype=np.uint8)
    if mode == 'exact':
        mask[image == level] = 1
    elif mode == 'above':
        mask[image > level] = 1
    else:
        mask[image < level] = 1

    return mask


##-


##-
def mask_to_external_contours(mask: np.ndarray, approx_factor: float = None, min_area: int = None) -> list:
    """Extract contours from a mask.

    Args:
        mask (numpy.array): a binary image
        approx_factor (float): if provided, the contours are simplified by the
            given factor (see cv2.approxPolyDP() function)
        min_area (float): if provided, filters out contours (polygons) with an area
            less than this value

    Returns:
        a list of contours (shapely.Polygon)
    """
    m = np.pad(mask.astype(np.uint8), pad_width=2, mode="constant", constant_values=0)
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if approx_factor is not None:
        # approximate the contours by polygons. If the resulting polygon-like contour is
        # not a true polygon, increase the approximation factor:
        cnt_tmp = list()
        for c in cnt:
            while True:
                ct = cv2.approxPolyDP(c, approx_factor * cv2.arcLength(c, True), True)
                ct = ct.squeeze() - 2
                if len(ct) >= 3:
                    ct = Polygon(ct)
                else:
                    break
                if ct.geom.is_valid:
                    # valid polygon, add to the list
                    cnt_tmp.append(ct)
                    break
                else:
                    # invalid polygon, increase approximation factor
                    approx_factor *= 5
        cnt = cnt_tmp
    else:
        # remove eventual singleton dimensions and convert to Polygon (removing the padding of 2)
        cnt = [c.squeeze() - 2 for c in cnt]
        for c in cnt:
            c[c < 0] = 0

        cnt = [Polygon(c) for c in cnt if len(c) >= 3]

    if min_area is not None:
        res = [p for p in cnt if p.geom.area >= min_area]
        return res

    return cnt


##-


##-
def add_region(mask: np.ndarray, poly_line: np.ndarray) -> np.ndarray:
    """Add a new masking region by setting to 1 all the
    pixels within the boundaries of a polygon. The changes are
    operated directly in the array.

    Args:
        mask (numpy.array): an array possibly already containing
            some masked regions, to be updated
        poly_line (numpy.array): an N x 2 array with the (x,y)
            coordinates of the polygon vertices as rows

    Returns:
        a numpy.array - the updated mask
    """

    c, r = masked_points(poly_line, mask.shape)
    mask[r, c] = 1

    return mask


##-


##-
def masked_points(poly_line: np.ndarray, shape: tuple) -> tuple:
    """Compute the coordinates of the points that are inside the polygonal
    region defined by the vertices of the polygon.

    Args:
        poly_line (numpy.array): an N x 2 array with the (x,y)
            coordinates of the polygon vertices as rows
        shape (pair): the extend (width, height) of the rectangular region
            within which the polygon lies (typically image.shape[:2])
    Returns:
        a pair of lists (X, Y) where (X[i], Y[i]) are the coordinates of a
        point within the mask (polygonal region)
    """

    # check the last point to match the first one
    if not np.all(poly_line[0,] == poly_line[-1,]):
        poly_line = np.concatenate((poly_line, [poly_line[0,]]))

    # remember: row, col in polygon()
    r, c = skimage.draw.polygon(poly_line[:, 1], poly_line[:, 0], shape)

    return c, r


##-


##-
def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a mask to each channel of an image. Pixels corresponding to 0s in
    the mask will be set to 0. Changes are made in situ.

    Args:
        img (numpy.array): an image as an N-dim array
            (height x width x no_of_channels)
        mask (numpy.array): a mask as a 2-dim array (height x width)

    Return:
        numpy.array: the modified image
    """
    if mask.dtype is np.bool:
        mask = mask.astype(np.uint8)
    mask[mask > 0] = 1

    if img.ndim == 2:
        img *= mask
    else:
        for k in np.arange(img.shape[2]):
            img[:, :, k] *= mask

    return img
##-

##
## BinaryMask - single channel pyramidal image with storage.
##
class BinaryMask(object):
    def __init__(self, shape: dict[str, int],
                 dtype=np.uint8,
                 tmp_folder: str='/tmp',
                 tmp_prefix: str='tmp_binarymask_'):
        """Initialize a binary mask.

        Args:
            shape: (dict) {'width', 'height'} of the mask
            dtype: (np.dtype) type of values stored
            tmp_folder: (str) path where the temp mask is stored
            tmp_prefix: (str) prefix for the temp mask
        """
        self._shape = {'width' : shape['width'], 'height' : shape['height']}
        self._dtype = dtype
        self._white = np.iinfo(self._dtype).max
        if not Path(tmp_folder).exists():
            tmp_folder = '/tmp'
        random_file_name = tmp_prefix + md5(str(localtime()).encode('utf-8')).hexdigest() + '.zarr'
        self._mask_storage_path = Path(tmp_folder) / Path(random_file_name)

        chunks = (min(4096, self._shape['height']), min(4096, self._shape['width']))
        self._mask_storage = zarr.open(str(self._mask_storage_path), mode='w',
                                       shape=(self._shape['height'], self._shape['width']),
                                       chunks=chunks,
                                       dtype=self._dtype)
        self._mask_storage[:] = 0

        return

    def __del__(self):
        self._mask_storage.store.close()
        if self._mask_storage_path.exists():
            shutil.rmtree(self._mask_storage_path)
        return

    def get_temp_path(self) -> Path:
        return self._mask_storage_path

    @property
    def mask(self) -> zarr.Array:
        return self._mask_storage

    def to_image(self, dst_path: Path):
        self._mask_storage.set_mask_selection(self._mask_storage[:] > 0,
                                              self._white) # 0 - black, everything else - white
        tif.imwrite(dst_path.with_suffix('.tiff'), self._mask_storage,
                    bigtiff=True, photometric='minisblack',
                    compression='zlib', metadata={'axes': 'CYX'})
        return

    def to_pyramid(self, dst_path: Path,
                   current_level: int,
                   max_level: int,
                   min_level: int=0
                   ):
        """Generate a multi-resolution pyramid from the current mask.

        The pyramid will have the current mask as its <current_level> image, all
        other levels being generated via down-/up-sampling from it. The number
        of levels to be generated is controlled via <min_level> (non-negative) and
        <max_level>.

        Args:
            dst_path: (Path) where to save the pyramid.zarr dataset
            current_level: (int) the level in the pyramid represented by the mask
            max_level: (int) maximum level (smallest image) to be generated
            min_level: (int) minimum level (largest image) to be generated
        """


        min_level = max(min_level, 0)

        # now, find a max_level that does not shrink the image below 128px in either
        # width or height
        max_level = max(current_level, max_level)
        d = min(self._mask_storage.shape)
        l = max_level - current_level
        while d // 2**l < 128 and l >= 0:
            l -= 1
        if l < 0:
            l = 0
        max_level = current_level + l

        with zarr.open_group(str(dst_path.with_suffix('.zarr')), mode='w') as zroot:
            pyramid_info = []
            # up-sampling:
            for level in range(min_level, max_level):
                factor = 2**(current_level - level)
                mask_shape = (int(factor * self._mask_storage.shape[0]),
                              int(factor * self._mask_storage.shape[1]))
                chunks = (min(4096, mask_shape[0]), min(4096, mask_shape[1]))
                new_mask = zroot.create_dataset(str(level),
                                                shape=mask_shape,
                                                chunks=chunks,
                                                dtype=self._mask_storage.dtype)
                if level == current_level:
                    # just copy:
                    new_mask[:] = self._mask_storage[:]
                else:
                    new_mask[:] = resize(self._mask_storage[:], mask_shape, order=0,
                                         mode='reflect', anti_aliasing=False)

                new_mask.set_mask_selection(new_mask[:] > 0, self._white)

                pyramid_info.append({
                    'level': level,
                    'width': mask_shape[1],
                    'height' : mask_shape[0],
                    'downsample_factor': 2**(level - min_level)
                })
            zroot.attrs['pyramid'] = pyramid_info
            zroot.attrs['pyramid_desc'] = 'generated for binary mask'

        return
