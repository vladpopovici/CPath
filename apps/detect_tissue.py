# -*- coding: utf-8 -*-

# Detect tissue regions in a whole slide image (OME-TIFF).


#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################
from datetime import datetime
import hashlib

_time = datetime.now()
__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = "1.0"
__description__ = {
    'name': 'detect_tissue',
    'unique_id' : hashlib.md5(str.encode('detect_tissue' + __version__)).hexdigest(),
    'version': __version__,
    'timestamp': _time.isoformat(),
    'input': ['None'],
    'output': ['None'],
    'params': dict()
}

import simplejson as json
import geojson as gjson
import configargparse as opt
import numpy as np
from pathlib import Path

from shapely.affinity import translate

from cpath.core.wsi import WSIInfo
from cpath.core.mask import mask_to_external_contours
from cpath.proc.tissue_he import detect_foreground
from cpath.core import NumpyJSONEncoder
from cpath.core.mri import MRI
from cpath.core.annot import WSIAnnotation

# minimum object sizes (areas, in px^2) for different magnifications to be considered as "interesting"
# mpp -> min obj size
min_obj_size = {'15.0': 1500, '3.75': 50000, '1.8': 100000, '0.9': 500000}
WORK_MPP_1 = 15.0  # mpp
WORK_MPP_2 = 1.8   # mpp

def main():
    p = opt.ArgumentParser(description="Detect tissue regions in a whole slide image.")
    p.add_argument("--image", action="store", help="name of the whole slide image file (OME-TIFF format)",
                   required=True)
    p.add_argument("--out", action="store",
                   help="JSON file for storing the resulting annotation",
                   required=True)
    p.add_argument("--annotation_name", action="store", help="name of the resulting annotation",
                   default="tissue", required=False)
    p.add_argument("--min_area", action="store", type=int, default=None,
                   help="minimum area of a tissue region", required=False)
    p.add_argument("--he", action="store_true", help="use H&E-specific method for detecting the objects")

    args = p.parse_args()

    if args.min_area is None:
        args.min_area = min_obj_size[str(WORK_MPP_2)]
    else:
        min_obj_size[str(WORK_MPP_2)] = args.min_area

    in_path = Path(args.image).expanduser().absolute()
    out_path = Path(args.out).expanduser().absolute()
    __description__['params'] = vars(args)
    __description__['input'] = [str(in_path)]
    __description__['output'] = [str(out_path)]

    # print(__description__)

    wsi = WSIInfo(in_path)
    img_src = MRI(wsi)

    # use a two pass strategy: first detect a bounding box, then zoom-in and
    # detect the final mask
    level_1 = wsi.get_level_for_mpp(WORK_MPP_1)

    # print("Processing level: {}".format(level_1))

    img = img_src.get_plane(level=level_1)
    mask, _ = detect_foreground(img, method='fesi', min_area=min_obj_size[str(WORK_MPP_1)])
    contours = mask_to_external_contours(mask, approx_factor=0.0001, min_area=min_obj_size[str(WORK_MPP_1)])

    # find the bounding box of the contours:
    xmin, ymin = img.shape[:2]
    xmax, ymax = 0, 0
    for c in contours:
        minx, miny, maxx, maxy = c.geom.bounds
        xmin = min(xmin, minx)
        ymin = min(ymin, miny)
        xmax = max(xmax, maxx)
        ymax = max(ymax, maxy)

    # some free space around the ROI and rescale to new magnification level:
    level_2 = wsi.get_level_for_mpp(WORK_MPP_2)
    f = 2** int(abs(level_1 - level_2))
    xmin = int(f * max(0, xmin - 5))
    ymin = int(f * max(0, ymin - 5))
    xmax = int(f * min(img.shape[1] - 1, xmax + 5))
    ymax = int(f * min(img.shape[0] - 1, ymax + 5))

    # print("ROI @{}x: {},{} -> {},{}".format(WORK_MAG_2, xmin, ymin, xmax, ymax))
    img = img_src.get_region_px(xmin, ymin,
                                width=xmax - xmin, height=ymax - ymin,
                                level=level_2, as_type=np.uint8)
    # print("Image size 2: {}x{}".format(img.shape[0], img.shape[1]))

    if args.he:
        mask, _ = detect_foreground(img, method='simple-he', min_area=min_obj_size[str(WORK_MPP_2)])
    else:
        mask, _ = detect_foreground(img, method='fesi',
                                    #laplace_ker=15, gauss_ker=17, gauss_sigma=25.0,
                                    #morph_open_ker=5, morph_open_iter=7, morph_blur=17,
                                    laplace_ker=7, gauss_ker=7, gauss_sigma=9.0,
                                    morph_open_ker=5, morph_open_iter=7, morph_blur=11,
                                    min_area=min_obj_size[str(WORK_MPP_2)])

    contours = mask_to_external_contours(mask,
                                         approx_factor=0.00005,
                                         min_area=min_obj_size[str(WORK_MPP_2)])

    # don't forget to shift detections by (xmin, ymin) to obtain coords in original space for
    # this magnification level...
    for c in contours:
        if not c.geom.is_valid:
            raise RuntimeWarning("Contour {} is not valid".format(c.geom))
        c.geom = translate(c.geom, xoff=xmin, yoff=ymin)
        c._name = "tissue"
        c._in_group = "tissue_foreground"  

    # ...and get image extent at working magnification
    img_shape = img_src.extent(level_2)
    annot = WSIAnnotation(name=args.annotation_name, image_shape=img_shape, 
                          mpp=wsi.get_mpp_for_level(level_2),
                          group_list=['tissue_foreground'])  # use exact image's mpp
    annot.add_annotations(contours)

    # get back to native magnification...
    annot.set_mpp(wsi.get_native_resolution())
    # ...and correct the image extent (due to rounding it may be off by a few pixels), since
    # we actually know it:
    img_shape = img_src.extent(0)
    annot._image_shape = dict(width=img_shape[0], height=img_shape[1])

    with open(out_path, 'w') as f:
        tmp = annot.asGeoJSON()
        tmp['__description__'] = __description__
        gjson.dump(tmp, f, cls=NumpyJSONEncoder)

    return
##


if __name__ == '__main__':
    main()
