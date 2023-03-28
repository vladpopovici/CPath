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
    p.add_argument("--mpp", action="store", help="approximate work resolution (microns-per-pixel) (defaults to min mpp)", 
                   required=False, default=0.0, type=float)
    p.add_argument("--min_area", action="store", type=int, default=None,
                   help="minimum area of a nuclei", required=False)
    p.add_argument("--max_area", action="store", type=int, default=None,
                   help="maximum area of a nuclei", required=False)
    p.add_argument("--min_prob", action="store", type=float, default=0.5,
                   help="all candidate dections below this minimum probability will be discarded")

    args = p.parse_args()

    __description__['params'] = vars(args)

    in_path = pathlib.Path(args.image).expanduser().absolute()
    out_path = pathlib.Path(args.out).expanduser().absolute()

    __description__['input'] = [in_path]
    __description__['output'] = [out_path]

    args.min_prob = max(0, min(args.min_prob, 1.0))

    keras = keras_import()
    nrm = IntensityNormalizer(0, 255)
    wsi = WSIInfo(in_path)
    img_src = MRI(wsi)

    if args.mpp == 0.0:
        level = 0
    else:
        level = wsi.get_level_for_mpp(args.mpp)

    img_shape = wsi.get_extent_at_level(level)
    sz = min(math.floor(math.sqrt(min(img_shape['width'], img_shape['height'])))**2, 2*4096)
    sz = int(sz)

    model = StarDist2D.from_pretrained('2D_versatile_he')

    img = img_src.get_plane(level)
    _, polys = model.predict_instances_big(img, axes='YXC',
                                           block_size=sz,
                                           min_overlap=128, context=128,
                                           normalizer=nrm,
                                           n_tiles=(4, 4, 1),
                                           labels_out=False,
                                           show_progress=True)
    #with open("/home/vlad/tmp/nuclei.json" , 'w') as f:
    #    json.dump(polys, f, cls=NumpyJSONEncoder)

    (idx,) = np.where(np.array(polys['prob']) >= args.min_prob)
    n = len(polys['prob'])
    annot = WSIAnnotation('nuclei',
                          wsi.get_extent_at_level(level),
                          mpp=wsi.get_mpp_for_level(level))  # exact mpp
    for k in idx:
        p = Polygon([xy for xy in zip(polys['coord'][k][0], polys['coord'][k][1])])
        annot.add_annotation_object(p)

    with open(out_path, 'w') as f:
        gjson.dump(annot.asGeoJSON(), f, cls=NumpyJSONEncoder)

    return
##


if __name__ == '__main__':
    main()
