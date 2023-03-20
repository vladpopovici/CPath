#   -*- coding: utf-8 -*-
#
#  --------------------------------------------------------------------
#  Copyright (c) 2022-2023 Vlad Popovici <popovici@bioxlab.org>
#
#  Licensed under the MIT License. See LICENSE file in root folder.
#  --------------------------------------------------------------------

# MAGNIF: functions for handling resolution/magnification conversion.

__all__ = ['Magnification']


import numpy as np


class Magnification:
    def __init__(self, magnif: float, mpp: float, level: int=0,
                 n_levels: int=10, magnif_step: float=2.0):
        """Magnification handling/conversion.

        Args:
            magnif: objective magnification (e.g. 10.0)
            mpp: resolution in microns per pixel at the given magnification
                (e.g. 0.245)
            level: level in the image pyramid corresponding to the
                magnification/mpp. Defaults to 0 - highest magnification.
            n_levels: number of levels in the image pyramid that are relevant/
                feasible
            magnif_step: scaling factor between levels in the image pyramid.
        """
        if level < 0 or level >= n_levels:
            raise RuntimeError("Specified level outside [0, (n_levels-1)]")
        self._base_magnif = magnif
        self._base_mpp = mpp
        self._base_level = level
        self._magnif_step = magnif_step
        self._n_levels = n_levels
        # initialize look-up tables:
        self.__magnif = magnif * self._magnif_step ** (level - np.arange(n_levels))
        self.__mpp    = mpp * self._magnif_step ** (np.arange(n_levels) - level)

        return


    def get_magnif_for_mpp(self, mpp: float) -> float:
        """
        Returns the objective magnification for a given resolution.
        Args:
            mpp: target resolution

        Returns:
            float: magnification corresponding to mpp. If <mpp> is outside the
                normal interval then return the corresponding end of the
                magnification values if still close enough (relative error <=0.1)
                or raise an Error
        """
        if mpp < self.__mpp[0]:
            # mpp outside normal interval, try to see if it's too far:
            if (self.__mpp[0] - mpp) / self.__mpp[0] > 0.1:
                raise RuntimeError('mpp outside supported interval') from None
            else:
                return self.__magnif[0]
        if mpp > self.__mpp[self._n_levels-1]:
            # mpp outside normal interval, try to see if it's too far:
            if (mpp - self.__mpp[self._n_levels-1]) / self.__mpp[self._n_levels-1] > 0.1:
                raise RuntimeError('mpp outside supported interval') from None
            else:
                return self.__magnif[self._n_levels-1]
        k = np.argmin(np.abs(mpp - self.__mpp))

        return self.__magnif[k]

    def get_mpp_for_magnif(self, magnif: float) -> float:
        """
        Return the resolution (microns per pixel - mpp) for a given objective
            magnification.
        Args:
            magnif: target magnification

        Returns:
            float: resolution (mpp) corresponding to magnification
        """
        if magnif > self.__magnif[0] or magnif < self.__magnif[self._n_levels-1]:
            raise RuntimeError('magnif outside supported interval') from None
        k = np.argmin(np.abs(magnif - self.__magnif))

        return self.__mpp[k]

    def get_level_for_magnif(self, magnif: float) -> int:
        """
        Return the level for a given objective magnification. Negative values
        correspond to magnification levels higher than the indicated base level.

        Args:
            magnif: target magnification

        Returns:
            int: resolution (mpp) corresponding to magnification
        """
        if magnif > self.__magnif[0] or magnif < self.__magnif[self._n_levels - 1]:
            raise RuntimeError('magnif outside supported interval') from None

        k = np.argmin(np.abs(magnif - self.__magnif))

        return k


    def get_level_for_mpp(self, mpp: float) -> int:
        """
        Return the level for a given resolution. Negative values
        correspond to resolution levels higher than the indicated base level.

        Args:
            mpp: target resolution

        Returns:
            int: resolution (mpp) corresponding to magnification
        """
        if mpp < self.__mpp[0]:
            # mpp outside normal interval, try to see if it's too far:
            if (self.__mpp[0] - mpp) / self.__mpp[0] > 0.1:
                raise RuntimeError('mpp outside supported interval') from None
            else:
                return 0
        if mpp > self.__mpp[self._n_levels-1]:
            # mpp outside normal interval, try to see if it's too far:
            if (mpp - self.__mpp[self._n_levels-1]) / self.__mpp[self._n_levels-1] > 0.1:
                raise RuntimeError('mpp outside supported interval') from None
            else:
                return self._n_levels-1

        k = np.argmin(np.abs(mpp - self.__mpp))

        return k


    def get_mpp_for_level(self, level:int) -> float:
        """
        Return the resolution (mpp) for a given level.

        Args:
            level: target level

        Returns:
            float: resolution (mpp)
        """
        if level < 0 or level >= self._n_levels:
            raise RuntimeError('level outside supported interval.') from None

        return self.__mpp[level]
    
    def get_magnification_step(self) -> float:
        """
        Return the magnification step between two consecutive levels.

        Returns:
            float: magnification step
        """
        return self._magnif_step
    