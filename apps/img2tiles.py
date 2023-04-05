# -*- coding: utf-8 -*-

# Extract tiles from a polygonal region of an image (OME-TIFF).

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
    'name': 'img2tiles',
    'version': __version__,
    'timestamp': _time.isoformat(),
    'input': ['None'],
    'output': ['None'],
    'params': dict()
}

import configargparse as opt
from pathlib import Path
from cpath.core.mri import MRI
from cpath.core.sampler import SlidingWindowSampler
from cpath.core.wsi import WSIInfo
from cpath.core.annot import WSIAnnotation
import geojson
from shapely.ops import unary_union
import imageio
import pandas as pd


def main():
    p = opt.ArgumentParser(description="Split a region from a OME-TIFF file (image) into a set of tiles (.png).")
    p.add_argument("--image", action="store", help="name of the whole slide image file (OME-TIFF format)",
                   required=True)
    p.add_argument("--out", action="store", help="output folder", required=True)
    p.add_argument("--mpp", action="store", help="approximate work resolution (microns-per-pixel) (defaults to min mpp)", 
                   required=False, default=0.0, type=float)
    p.add_argument("--tile_shape", action="store", nargs=2, type=int, 
                   metavar=['W','H'],
                   required=False, default=(256,256),
                   help="tile shape (width, height)")
    p.add_argument("--tile_step", action="store", nargs=2, type=int, required=False, default=(256, 256),
                   metavar="Sx, Sy",
                   help="step (x_step, y_step) from one tile to the next")
    p.add_argument("--annot_file", action="store", default=None, required=False,
                   help="annotation with the region(s) of interest (union of polygons in the annotation)")
    p.add_argument("--annot_group", action="store", default="tissue_foreground",
                   help="name of annotation group")
    p.add_argument("--prefix", action="store", default="tile_",
                   help="resulting files will have the name <prefix><index>.png")


    args = p.parse_args()

    in_path = Path(args.image).expanduser().absolute()
    out_path = Path(args.out).expanduser().absolute()
    __description__['params'] = vars(args)
    __description__['input'] = [str(in_path)]
    __description__['output'] = [str(out_path)]

    # print(__description__)

    wsi = WSIInfo(in_path)
    img_src = MRI(wsi)

    if args.mpp == 0.0:
        level = 0
    else:
        level = wsi.get_level_for_mpp(args.mpp)

    level_shape = wsi.get_extent_at_level(level)
    level_mpp = wsi.get_mpp_for_level(level)  # exact mpp
    region = None

    if args.annot_file is not None:
        annot = WSIAnnotation('tissue', level_shape, mpp=level_mpp)
        with open(args.annot_file, 'r') as f:
            annot.fromGeoJSON(geojson.load(f))
        annot.set_mpp(level_mpp)
        # the tissue region may be formed by several pieces
        region = unary_union(
            [p.geom for p in annot._annots[args.annot_group]]
        )
        print(f"Tissue region bounding box: {region.bounds}")

    sampler = SlidingWindowSampler(
        image_shape=level_shape,
        w_size = {'width': args.tile_shape[0], 'height': args.tile_shape[1]},
        start=(0, 0),
        step=args.tile_step,
        poly=region,
        nv_inside=2,  # at least 2 corners of the tile must be in the region
        only_full_windows=True
    )
    bbox = sampler.bounding_box()
    print(f"Bounding box: {bbox}")
    image = img_src.get_region_px(x0=bbox[0], y0=bbox[1],
                                  width=bbox[2]-bbox[0]+1, height=bbox[3]-bbox[1]+1,
                                  level=level)
    
    # out path
    if not out_path.exists():
        out_path.mkdir(parents=True)

    coords = [p for p in sampler]  # generate all coords as we will use them twice
    tile_names: list = []
    k = 1
    for p in coords:
        patch = image[p[1]:p[3], p[0]:p[2], ...]
        name = args.prefix + '{:08d}'.format(k) + '.png'
        imageio.imsave(out_path / name, patch)
        tile_names.append(name)
        k += 1
    coords = pd.DataFrame(coords, index = tile_names,
                          columns = ['x0', 'y0', 'x1', 'y1'])
    coords.to_csv(out_path / 'index.csv', sep='\t')


    return


if __name__ == '__main__':
    main()