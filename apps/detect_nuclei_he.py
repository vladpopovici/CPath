# -*- coding: utf-8 -*-

# Detects nuclei in H&E images.

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import math
from datetime import datetime
import hashlib

_time = datetime.now()
__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__version__ = "1.0"
__description__ = {
    'name': 'nuclei_dectection',
    'unique_id' : hashlib.md5(str.encode('nuclei_detection' + __version__)).hexdigest(),
    'version': __version__,
    'timestamp': _time.isoformat(),
    'input': ['None'],
    'output': ['None'],
    'params': dict()
}

import simplejson as json
import geojson as gjson
import configargparse as opt
import pathlib
import numpy as np
from csbdeep.utils.tf import keras_import
from csbdeep.data import Normalizer, normalize_mi_ma

from shapely.affinity import translate

from stardist.models import StarDist2D

from cpath.core.wsi import WSIInfo
from cpath.core import NumpyJSONEncoder
from cpath.core.annot import WSIAnnotation, Polygon
from cpath.core.mri import MRI

class IntensityNormalizer(Normalizer):
    def __init__(self, mi=0, ma=255):
        self.mi, self.ma = mi, ma

    def before(self, x, axes):
        return normalize_mi_ma(x, self.mi, self.ma, dtype=np.float32)

    def after(*args, **kwargs):
        assert False

    @property
    def do_after(self):
        return False


def main():
    p = opt.ArgumentParser(description="Detect nuclei in an H&E slide image.")
    p.add_argument("--image", action="store", help="name of the whole slide image file (OME-TIFF format)",
                   required=True)
    p.add_argument("--out", action="store",
                   help="JSON file for storing the resulting annotation (nuclei positions)",
                   required=True)
    p.add_argument("--roi", action="store", help="annotation file with region(s) of interest", default=None, required=False)
    p.add_argument("--roi_name", action="store", help="annotation name", default="tissue", required=False)
    p.add_argument("--roi_group", action="store", help="annotation group to use", default="NO_GROUP", required=False)
    p.add_argument("--out_annot_name", action="store", default='nuclei_stardist', required=False)
    p.add_argument("--out_annot_group", action="store", default='nuclei', required=False)
    p.add_argument("--mpp", action="store", help="approximate work resolution (microns-per-pixel) (defaults to min mpp)", 
                   required=False, default=0.0, type=float)
    p.add_argument("--min_area", action="store", type=int, default=None,
                   help="minimum area of a nuclei", required=False)
    p.add_argument("--max_area", action="store", type=int, default=None,
                   help="maximum area of a nuclei", required=False)
    p.add_argument("--min_prob", action="store", type=float, default=0.5,
                   help="all candidate dections below this minimum probability will be discarded")
    p.add_argument("--show_progress", action="store_true", help="show progress bars")

    args = p.parse_args()

    __description__['params'] = vars(args)

    in_path = pathlib.Path(args.image).expanduser().absolute()
    out_path = pathlib.Path(args.out).expanduser().absolute()

    __description__['input'] = [str(in_path)]
    __description__['output'] = [str(out_path)]

    print(__description__)
    args.min_prob = max(0, min(args.min_prob, 1.0))

    keras = keras_import()
    nrm = IntensityNormalizer(0, 255)
    wsi = WSIInfo(in_path)
    img_src = MRI(wsi)

    if args.mpp == 0.0:
        level = 0
    else:
        level = wsi.get_level_for_mpp(args.mpp)

    model = StarDist2D.from_pretrained('2D_versatile_he')

    # resulting detections are saved as annotations in 'annot'
    annot = WSIAnnotation(args.out_annot_name,
                          wsi.get_extent_at_level(level),
                          mpp=wsi.get_mpp_for_level(level),  # exact mpp
                          group_list=[args.out_annot_group]) 

    if args.roi is None:
        # run on whole image
        img_shape = wsi.get_extent_at_level(level)
        sz = min(math.floor(math.sqrt(min(img_shape['width'], img_shape['height'])))**2, 2*4096)
        sz = int(sz)

        img = img_src.get_plane(level)
        _, polys = model.predict_instances_big(img, axes='YXC',
                                            block_size=sz,
                                            min_overlap=128, context=128,
                                            normalizer=nrm,
                                            n_tiles=(4, 4, 1),
                                            labels_out=False,
                                            show_progress=args.show_progress)
        #with open("/home/vlad/tmp/nuclei.json" , 'w') as f:
        #    json.dump(polys, f, cls=NumpyJSONEncoder)
        (idx,) = np.where(np.array(polys['prob']) >= args.min_prob)
        n = 1
        for k in idx:
            p = Polygon([xy for xy in zip(polys['coord'][k][0], polys['coord'][k][1])])
            annot.add_annotation_object(p, name=args.out_annot_name+f'_{n}', in_group=args.out_annot_group)
            n += 1
    else:
        # run on parts, masked
        roi = WSIAnnotation("tissue", (0,0), mpp=0) # dummy annotation, read it from file:
        with open(args.roi, 'r') as f:
            roi.fromGeoJSON(json.load(f))
        if roi.name != args.roi_name:
            raise RuntimeError("Annotation name not found in the given file")
        roi.set_mpp(wsi.get_mpp_for_level(level))  # set the resolution of the annotation to match WSI
        roi_parts = roi.get_group(args.roi_group)
        all_detections = list()
        n = 1
        for r in roi_parts:
            # this is too slow - use just the bounding box for the moment:
            #img = img_src.get_polygonal_region_px(r.geom, level)
                       
            xmin, ymin, xmax, ymax = [int(_z) for _z in r.geom.bounds]
            img = img_src.get_region_px(xmin, ymin, xmax-xmin, ymax-ymin, level)
            
            img_shape = {'width': img.shape[1], 'height': img.shape[0]}
            sz = min(math.floor(math.sqrt(min(img_shape['width'], img_shape['height'])))**2, 2*4096)
            sz = int(sz)

            _, polys = model.predict_instances_big(img, axes='YXC', block_size=sz,
                                                   min_overlap=128, context=128,
                                                   normalizer=nrm, n_tiles=(4, 4, 1),
                                                   labels_out=False, show_progress=args.show_progress)
            (idx,) = np.where(np.array(polys['prob']) >= args.min_prob)  # detections above threshold
            for k in idx:
                p = Polygon(
                    [xy for xy in zip(polys['coord'][k][0], polys['coord'][k][1])],
                    name=args.out_annot_name + f'_{n}',
                    in_group=args.out_annot_group
                    )
                # don't forget to shift detections by (xmin, ymin) to obtain coords in original space for
                # this magnification level...
                p.geom = translate(p.geom, xoff=xmin, yoff=ymin)
                n += 1
                all_detections.append(p)

        annot.add_annotations(all_detections)
        

    tmp = annot.asGeoJSON()
    tmp['__description__'] = __description__

    with open(out_path, 'w') as f:
        gjson.dump(tmp, f, cls=NumpyJSONEncoder)

    return
##


if __name__ == '__main__':
    main()
