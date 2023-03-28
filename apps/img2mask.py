#
# Convert an image into a binary mask (values 0, 1).
#

import configargparse as opt
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from pathlib import Path



def main():
    p = opt.ArgumentParser(description="Convert an image into a binary mask (.bmp, {0,1}-valued).")
    p.add_argument("--input", action="store", help="input file name", required=True)
    p.add_argument("--output", action="store", help="output file (.bmp)", required=True)
    p.add_argument("--cutoff", action="store", default=(128,128,128), required=False, nargs=3,
                   help="cuttoffs for each channel (3 values)")

    args = p.parse_args()

    img0 = imread(args.input)
    if img0.ndim > 2:
        mask = (img0[..., 0] >= args.cuttoff[0]) * \
               (img0[..., 1] >= args.cuttoff[1]) * \
               (img0[..., 2] >= args.cuttoff[2])
    else:
        mask = img0 >= args.cutoff[0]

    imsave(Path(args.output).with_suffix('.bmp'), img_as_ubyte(mask)-255, check_contrast=False)


    return


if __name__ == "__main__":
    main()
