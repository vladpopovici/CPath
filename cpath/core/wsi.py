#   -*- coding: utf-8 -*-
#
#  --------------------------------------------------------------------
#  Copyright (c) 2022-2023 Vlad Popovici <popovici@bioxlab.org>
#
#  Licensed under the MIT License. See LICENSE file in root folder.
#  --------------------------------------------------------------------

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = "0.3.5"

"""
WSI: Defines WSIInfo class.
"""

__all__ = ['WSIInfo', 'NumpyImage']

import pathlib
import numpy as np
from math import log
import zarr
import re
from typing import Optional
import pyometiff
import tifffile

from . import ImageShape
from .magnif import Magnification


#####
class WSIInfo(object):
    """Hold some basic info about a WSI. This metadata is structured after OME XML format.
    The whole slide image is supposed to be in OME-TIFF format (such as produced by
    bioformats2raw + raw2ometiff tools). Note that while there might be several images 
    stored in the file, just the first one is used.

    Args:
        path (str): full path to the image file (.zarr.zip).

    Attributes:
        path (str): full path to WSI file
        info (dict): see the OME-XML and OME-ZARR specifications.
    """

    def __init__(self, path: str):
        tif = tifffile.TiffFile(path)
        if not tif.is_ome:
            raise RuntimeError("only OME TIFF files are accepted.")
        
        self.path = pathlib.Path(path)

        ome = pyometiff.OMETIFFReader(fpath=path)
        ome.omexml_string = tif.ome_metadata  # work-around a bug in pyometiff
        self.info = ome.parse_metadata(tif.ome_metadata)
        
        # axes: identify singletons and remove from axes annotation
        self.axes = ''.join(
            [a for a in list(self.info['DimOrder']) if self.info['Size' + a] > 1]
        ).lower() # -> something like 'cyx'

        # axes of interest
        self.axes_order = {
            'c': list(self.axes).index('c'), 
            'y': list(self.axes).index('y'),
            'x': list(self.axes).index('x'),
            } # just X, Y, C

        self._pyramid_levels = dict()  # convenient access to pyramid levels info
        
        tif = tifffile.imread(self.path, aszarr=True)
        if not tif.is_multiscales:
            raise RuntimeError("only pyramidal images are accepted.")

        if self.info['PhysicalSizeXUnit'] == 'µm':
            unit_multiplier_x = 1.0 
        elif self.info['PhysicalSizeXUnit'] == 'mm': 
            unit_multiplier_x = 1000.0
        else:
            unit_multiplier_x = 1.0
            raise RuntimeWarning('unknown unit for resolution (X)')
        
        if self.info['PhysicalSizeYUnit'] == 'µm':
            unit_multiplier_y = 1.0 
        elif self.info['PhysicalSizeYUnit'] == 'mm': 
            unit_multiplier_y = 1000.0
        else:
            unit_multiplier_y = 1.0
            raise RuntimeWarning('unknown unit for resolution (Y)')

        base_mpp_x = unit_multiplier_x * self.info['PhysicalSizeX']  # in microns per pixel
        base_mpp_y = unit_multiplier_y * self.info['PhysicalSizeY']

        with zarr.open(tif, mode='r') as z:
            lv_labels = list(z.array_keys())
            k = 0
            for lv in lv_labels:
                dims = z[lv].shape
                self._pyramid_levels[int(lv)] = {
                    'path': lv,
                    'mpp_x': 1.0,
                    'mpp_y': 1.0,
                    'shape': dims,
                    'max_x': dims[self.axes_order['x']],
                    'max_y': dims[self.axes_order['y']]
                }
                fx = self._pyramid_levels[0]['max_x'] / self._pyramid_levels[int(lv)]['max_x']
                fy = self._pyramid_levels[0]['max_y'] / self._pyramid_levels[int(lv)]['max_y']
                self._pyramid_levels[int(lv)]['mpp_x'] = fx * base_mpp_x
                self._pyramid_levels[int(lv)]['mpp_y'] = fy * base_mpp_y

        self.info['objective_power'] = float(self.info['ObjMag'])

        # get the scaling factor between levels:
        if len(self._pyramid_levels) > 1:
            ms_x = self._pyramid_levels[0]['max_x'] / self._pyramid_levels[1]['max_x']
            ms_y = self._pyramid_levels[0]['max_y'] / self._pyramid_levels[1]['max_y']
            ms = round(0.5*(ms_x + ms_y))
        else:
            ms = 1.0
        self.magnif_converter =  Magnification(
            self.info['objective_power'],
            mpp=0.5*(self._pyramid_levels[0]['mpp_x']+self._pyramid_levels[0]['mpp_y']),
            level=0,
            magnif_step=float(ms))

        return

    def ix(self) -> int:
        """Return the X-axis index"""
        return self.axes_order['x']
    
    def iy(self) -> int:
        """Return the Y-axis index"""
        return self.axes_order['y']
        
    def ic(self) -> int:
        """Return the C-axis (channels) index"""
        return self.axes_order['c']
    
    def level_count(self) -> int:
        """Return the number of levels in the multi-resolution pyramid."""
        return len(self._pyramid_levels)


    def downsample_factor(self, level:int) -> int:
        """Return the downsampling factor (relative to level 0) for a given level."""
        if level < 0 or level >= self.level_count():
            return -1
        ms_x = self._pyramid_levels[level]['mpp_x'] / self._pyramid_levels[0]['mpp_x']
        ms_y = self._pyramid_levels[level]['mpp_y'] / self._pyramid_levels[0]['mpp_y']
        ms = round(0.5*(ms_x + ms_y))
        return float(ms)

    def get_native_magnification(self) -> float:
        """Return the original magnification for the scan."""
        return self.info['objective_power']


    def get_native_resolution(self) -> float:
        """Return the scan resolution (microns per pixel)."""
        return 0.5 * (self._pyramid_levels[0]['mpp_x'] + self._pyramid_levels[0]['mpp_y'])


    def get_level_for_magnification(self, mag: float, eps=1e-6) -> int:
        """Returns the level in the image pyramid that corresponds the given magnification.

        Args:
            mag (float): magnification
            eps (float): accepted error when approximating the level

        Returns:
            level (int) or -1 if no suitable level was found
        """
        if mag > self.info['objective_power'] or mag < 2.0**(1-self.level_count()) * self.info['objective_power']:
            return -1

        lx = log(self.info['objective_power'] / mag, self.magnif_converter._magnif_step)
        k = np.where(np.isclose(lx, range(0, self.level_count()), atol=eps))[0]
        if len(k) > 0:
            return k[0]   # first index matching
        else:
            return -1   # no match close enough


    def get_level_for_mpp(self, mpp: float):
        """Return the level in the image pyramid that corresponds to a given resolution."""
        return self.magnif_converter.get_level_for_mpp(mpp)


    def get_mpp_for_level(self, level: int):
        """Return resolotion (mpp) for a given level in pyramid."""
        return self.magnif_converter.get_mpp_for_level(level)


    def get_magnification_for_level(self, level: int) -> float:
        """Returns the magnification (objective power) for a given level.

        Args:
            level (int): level in the pyramidal image

        Returns:
            magnification (float)
            If the level is out of bounds, returns -1.0
        """
        if level < 0 or level >= self.level_count():
            return -1.0
        if level == 0:
            return self.info['objective_power']

        #return 2.0**(-level) * self.info['objective_power']
        return self.magnif_converter._magnif_step ** (-level) * self.info['objective_power']


    def get_extent_at_level(self, level: int) -> Optional[ImageShape]:
        """Returns width and height of the image at a desired level.

        Args:
            level (int): level in the pyramidal image

        Returns:
            (width, height) of the level
        """
        if level < 0 or level >= self.level_count():
            return None
        return {'width': self._pyramid_levels[level]['max_x'],
                'height': self._pyramid_levels[level]['max_y']}


