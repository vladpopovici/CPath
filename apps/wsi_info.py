# -*- coding: utf-8 -*-

# Displays information about a whole slide image (OME-TIFF).

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

from datetime import datetime

_time = datetime.now()

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = "1.0"
__description__ = {
    'name': 'wsi_info',
    'version': __version__,
    'timestamp': _time.isoformat(),
    'input': ['None'],
    'output': ['None'],
    'params': dict()
}

import configargparse as opt
from pathlib import Path
import geojson as gjson
from cpath.core.wsi import WSIInfo
from cpath.core import NumpyJSONEncoder


def main():
    show_choices = ['nlevels', 'mpp', 'magnification', 'shape', 'pyramid']
    p = opt.ArgumentParser(description="Show information about a whole slide image (OME-TIFF).")
    p.add_argument("--image", action="store", help="name of the whole slide image file (OME-TIFF format)",
                   required=True)
    p.add_argument("--out", action="store", help="output file (JSON)", required=False, default=None)
    p.add_argument("--show", action="store", type=str, metavar='WHAT',
                   choices=['all', 'none'] + show_choices, nargs='+', default='all', required=True,
                   help="what to show: all, " + ', '.join(show_choices))

    args = p.parse_args()

    in_path = Path(args.image).expanduser().absolute()
    out_path = None if args.out is None else Path(args.out).expanduser().absolute()

    __description__['params'] = vars(args)
    __description__['input'] = [str(in_path)]
    __description__['output'] = [str(out_path)]


    to_show = []
    for show in args.show:
        if show.lower() == 'all':
            to_show = show_choices
            break
        if show.lower() == 'none':
            to_show = []
            break
        to_show.append(show.lower())
    
    if len(to_show) == 0 and out_path is None:
        # nothing to show, no file to save to...
        return
    
    wsi = WSIInfo(in_path)

    info = {
        'nlevels': wsi.level_count(),
        'mpp': wsi.get_native_resolution(),
        'magnification': wsi.get_native_magnification(),
        'shape': wsi.get_extent_at_level(0),
        'pyramid': [wsi.get_extent_at_level(k) for k in range(wsi.level_count())]
    }

    for show in to_show:
        print(info[show])

    if out_path is not None:
        info['__description__'] = __description__
        with open(out_path, 'w') as f:
            gjson.dump(info, f, cls=NumpyJSONEncoder)

    return


if __name__ == "__main__":
    main()
