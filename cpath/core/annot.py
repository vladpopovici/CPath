# -*- coding: utf-8 -*-

#############################################################################
# Copyright Vlad Popovici <popovici@bioxlab.org>
#
# Licensed under the MIT License. See LICENSE file in root folder.
#############################################################################

"""
ANNOT module for handling complex annotations and importing from other
systems (ASAP, NDPI,...).
"""

__author__ = "Vlad Popovici <popovici@bioxlab.org>"
__all__ = [
    'AnnotationObject',
    'Point',
    'PointSet',
    'PolyLine',
    'Polygon',
    'WSIAnnotation',
    'annotation_from_ASAP'
]

from abc import ABC, abstractmethod
from typing import Tuple, Union

import geojson as gj
import numpy as np
import shapely.affinity as sha
import shapely.geometry as shg
from pathlib import Path
import xmltodict
import csv


##-
class AnnotationObject(ABC):
    """Define the AnnotationObject's minimal interface. This class is made
    abstract to force more meaningful names (e.g. Point, Polygon, etc.) in
    subclasses."""

    def __init__(self, name: str = None, in_group: str = "NO_GROUP"):
        super().__init__()

        self._name = name
        self._annotation_type = None
        # main geometrical object describing the annotation:
        self.geom = shg.base.BaseGeometry()
        self._in_group = "NO_GROUP" if in_group is None else in_group

    @abstractmethod
    def duplicate(self):
        pass

    @staticmethod
    def empty_object():
        """Return an empty object to be initialized later"""
        pass

    @property
    def in_group(self):
        return self._in_group

    def __str__(self):
        """Return a string representation of the object."""
        return str(self.type) + " <" + str(self.name) + ">: " + str(self.geom)

    def bounding_box(self):
        """Compute the bounding box of the object."""
        return self.geom.bounds

    def translate(self, x_off, y_off=None):
        """Translate the object by a vector [x_off, y_off], i.e.
        the new coordinates will be x' = x + x_off, y' = y + y_off.
        If y_off is None, then the same value as in x_off will be
        used.
        
        :param x_off: (double) shift in thr X-direction
        :param y_off: (double) shift in the Y-direction; if None,
            y_off == x_off
        """
        if y_off is None:
            y_off = x_off
        self.geom = sha.translate(self.geom, x_off, y_off, zoff=0.0)

        return

    def scale(self, x_scale, y_scale=None, origin='center'):
        """Scale the object by a specified factor with respect to a specified
        origin of the transformation. See shapely.geometry.scale() for details.

        :param x_scale: (double) X-scale factor
        :param y_scale: (double) Y-scale factor; if None, y_scale == x_scale
        :param origin: reference point for scaling. Default: "center" (of the
            object). Alternatives: "centroid" or a shapely.geometry.Point object
            for arbitrary origin.
        """
        if y_scale is None:
            y_scale = x_scale
        self.geom = sha.scale(self.geom, xfact=x_scale, yfact=y_scale, zfact=1, origin=origin)

        return

    def resize(self, factor: float) -> None:
        """Resize an object with the specified factor. This is equivalent to
        scaling with the origin set to (0,0) and same factor for both x and y 
        coordinates.

        :param factor: (float) resizig factor.
        """
        self.scale(factor, origin=shg.Point((0.0, 0.0)))

        return

    def affine(self, M):
        """Apply an affine transformation to all points of the annotation.

        If M is the affine transformation matrix, the new coordinates
        (x', y') of a point (x, y) will be computed as

        x' = M[1,1] x + M[1,2] y + M[1,3]
        y' = M[2,1] x + M[2,2] y + M[2,3]

        In other words, if P is the 3 x n matrix of n points,
        P = [x; y; 1]
        then the new matrix Q is given by
        Q = M * P

        :param M: numpy array [2 x 3]

        :return:
            nothing
        """

        self.geom = sha.affine_transform(self.geom, [M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[0, 2], M[1, 2]])

        return

    @property
    def x(self) -> np.array:
        """Return the x coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return np.array(self.geom.x)

    @property
    def y(self) -> np.array:
        """Return the y coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return np.array(self.geom.y)

    @property
    def xy(self) -> np.array:
        """Return the xy-coordinates as a numpy.array [n,2]"""
        x, y = self.geom.coords.xy
        return np.array((x, y)).T

    @property
    def size(self) -> int:
        """Return the number of points defining the object."""
        return 0

    def asdict(self) -> dict:
        """Return a dictionary representation of the object."""
        d = {
            "annotation_type": self._annotation_type,
            "name": self._name,
            "in_group": self._in_group,
            "x": self.x,
            "y": self.y
        }

        return d

    def fromdict(self, d: dict) -> None:
        """Intialize the objct from a dictionary."""

        self._annotation_type = d["annotation_type"]
        self._name = d["name"]
        self._in_group = d["in_group"]

    def asGeoJSON(self) -> dict:
        """Return a dictionary compatible with GeoJSON specifications."""
        return gj.Feature(geometry=shg.mapping(self.geom),
                          properties=dict(object_type="annotation",
                                          annotation_type=self._annotation_type,
                                          name=self._name,
                                          in_group=self._in_group)
                          )

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        """This is a basic function - further tests should be implemented for particular object
        types."""

        self.geom = shg.shape(d["geometry"])
        try:
            self._name = d["properties"]["name"]
            self._in_group = d["properties"]["in_group"]
            self._annotation_type = d["properties"]["annotation_type"]
        except KeyError:
            pass

    @property
    def name(self):
        """Return the name of the annotation object."""
        return self._name

    @property
    def type(self):
        """Return the annotation type as a string."""
        return self._annotation_type


##-


class WSIAnnotation(object):
    """
    An annotation is a list of AnnotationObjects eventually grouped into AnnotationGroups.
    The coordinates are in image pixel coordinates at the specified resolution (microns
    per pixel - mpp).
    """

    def __init__(self, name: str, image_shape: Union[dict, Tuple[int, int]],
                 mpp: float, group_list: list[str] = []) -> None:
        """Initialize an Annotation for a slide.

        :param name: (str) name of the annotation
        :param image_shape: dict or (int, int) image shape (width, height). If dict,
            'width' and 'height' keys must be used.
        :param mpp: resolution at which the annotation was created (mpp relates
            to objectve power)
        """
        self._name = name
        if image_shape is tuple:
            self._image_shape = {'width': image_shape[0], 'height': image_shape[1]}
        else:
            self._image_shape = {'width': image_shape['width'], 'height': image_shape['height']}
        self._mpp = mpp

        # The annotations are stored in a dict(), indexed by group names.
        # If an annotation does not belong to any group, it is assigned to
        # "NO_GROUP" entry.
        self._annots = {'NO_GROUP': list()}  # dict with annotation objects
        if len(group_list) > 0:
            # initialize the groups, aside from NO_GROUP
            for g in group_list:
                self._annots[g] = list()

        return

    def add_annotation_object(self, a: AnnotationObject) -> None:
        # An annotation object must belong to a Group, be it NO_GROUP
        if a.in_group in self._annots:
            self._annots[a.in_group].append(a)
        else:
            self._annots[a.in_group] = [a]
        return

    def add_annotations(self, a: list) -> None:
        for o in a:
            self.add_annotation_object(o)  # takes care of groups as well

    def get_image_shape(self) -> dict:
        return self._image_shape

    @property
    def name(self):
        """Return the name of the annotation object."""
        return self._name

    @property
    def type(self):
        """Return the annotation type as a string."""
        return 'WSIAnnotation'

    def get_mpp(self) -> float:
        return self._mpp

    def resize(self, factor: float) -> None:
        self._mpp != factor  # mpp varies inverse proportional with scaling of the objects
        self._image_shape = {'width': self._image_shape['width'] * factor,
                             'height': self._image_shape['height'] * factor}

        for g in list(self._annots):  # for all groups
            for a in self._annots[g]:  # and all annotations in a group
                a.resize(factor)

        return

    def set_mpp(self, mpp: float) -> None:
        """Scales the annotation to the desired magnification.

        :param magnfication: (float) target magnification
        """
        if mpp != self._mpp:
            f = self._mpp / mpp
            self.resize(f)
            self._mpp = mpp

        return

    def asdict(self) -> dict:
        d = {'name': self._name,
             'image_shape': self._image_shape,
             'mpp': self._mpp,
             'annotations': self._annots
             }

        return d

    def fromdict(self, d: dict) -> None:
        self._name = d['name']
        self._image_shape = d['image_shape']
        self._mpp = d['mpp']
        self._annots.clear()
        self._annots = d['annotations']

        return

    def asGeoJSON(self) -> dict:
        """Creates a dictionary compliant with GeoJSON specifications."""

        # GeoJSON does not allow for FeatureCollection properties, therefore
        # we save magnification and image extent as properties of individual
        # features/annotation objects.

        # Not all groups are saved: those empty are lost.

        all_annots = []
        for group in self._annots:
            for a in self._annots[group]:
                b = a.asGeoJSON()
                b["properties"]["resolution_mpp"] = self._mpp
                b["properties"]["image_shape"] = self._image_shape
                all_annots.append(b)

        return gj.FeatureCollection(all_annots)

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize an annotation from a dictionary compatible with GeoJSON specifications."""
        if d["type"].lower() != "featurecollection":
            raise RuntimeError("Need a FeatureCollection as annotation! Got: " + d["type"])

        self._annots.clear()
        mg, im_shape = None, None
        for a in d["features"]:
            obj = WSIAnnotation._createEmptyAnnotationObject(a["geometry"]["type"])
            obj.fromGeoJSON(a)
            self.add_annotation_object(obj)
            if mg is None and "properties" in a:
                mg = a["properties"]["resolution_mpp"]
            if im_shape is None and "properties" in a:
                im_shape = a["properties"]["image_shape"]
        self._mpp = mg
        self._image_shape = im_shape

        return

    @staticmethod
    def _createEmptyAnnotationObject(annot_type: str) -> AnnotationObject:
        """Function to create an empty annotation object of a desired type.

        Args:
            annot_type (str):
                type of the annotation object:
                DOT/POINT
                POINTSET/MULTIPOINT
                POLYLINE/LINESTRING
                POLYGON

        """
        obj = None
        if annot_type.upper() == 'DOT' or annot_type.upper() == 'POINT':
            obj = Point.empty_object()
        elif annot_type.upper() == 'POINTSET' or annot_type.upper() == 'MULTIPOINT':
            obj = PointSet.empty_object()
        elif annot_type.upper() == 'LINESTRING' or annot_type.upper() == 'POLYLINE':
            obj = PolyLine.empty_object()
        elif annot_type.upper() == 'POLYGON':
            obj = Polygon.empty_object()
        else:
            raise RuntimeError("unknown annotation type: " + annot_type)
        return obj


##-

##-
class Point(AnnotationObject):
    """Pointt: a single position in the image."""

    def __init__(self, coords=[0.0, 0.0], name=None, in_group="NO_GROUP"):
        """Initialize a DOT annotation, i.e. a single point in plane.

        Args:
            coords (list or vector or tuple): the (x,y) coordinates of the point
            name (str): the name of the annotation
        """
        super().__init__(name, in_group)
        self._annotation_type = "POINT"
        self.geom = shg.Point(coords)

        return

    def duplicate(self):
        return Point([self.x[0], self.y[0]], name=self.name, in_group=self._in_group)

    @staticmethod
    def empty_object():
        p = Point()
        return p

    @property
    def size(self) -> int:
        """Return the number of points defining the object."""
        return 1

    def fromdict(self, d: dict) -> None:
        """Intialize the objct from a dictionary."""
        super().fromdict(d)

        self.geom = shg.Point((d["x"], d["y"]))

        return

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        if d["geometry"]["type"].lower() != "point":
            raise RuntimeError("Need a Point feature! Got: " + str(d))

        super().fromGeoJSON(d)
        self._annotation_type = "POINT"

        return


##-


##-
class PointSet(AnnotationObject):
    """PointSet: an ordered collection of points."""

    def __init__(self, coords, name=None, in_group="NO_GROUP"):
        """Initialize a POINTSET annotation, i.e. a collection
         of points in plane.

        Args:
            coords (list or tuple): coordinates of the points as in [(x0,y0), (x1,y1), ...]
            name (str): the name of the annotation
        """
        super().__init__(name, in_group)
        self._annotation_type = "POINTSET"

        self.geom = shg.MultiPoint(coords)

        return

    def duplicate(self):
        ps = PointSet(self.geom.xy(), self._name, self._in_group)
        return ps

    @staticmethod
    def empty_object():
        p = PointSet([[0, 0]])
        return p

    @property
    def x(self) -> np.array:
        """Return the x coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return np.array([p.x for p in self.geom.geoms])

    @property
    def y(self) -> np.array:
        """Return the y coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        return np.array([p.y for p in self.geom.geoms])

    @property
    def xy(self) -> np.array:
        """Return the xy-coordinates as a numpy.array [n,2]"""

        return np.array((self.x, self.y)).T

    @property
    def size(self) -> int:
        """Return the number of points defining the object."""
        return self.x.size

    def fromdict(self, d: dict) -> None:
        """Intialize the objct from a dictionary."""
        super().fromdict(d)

        self.geom = shg.MultiPoint(np.array((d['x'], d['y'])).T)

        return

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        if d["geometry"]["type"].lower() not in ["pointset", "multipoint"]:
            raise RuntimeError("Need a MultiPoint feature! Got: " + str(d))

        super().fromGeoJSON(d)
        self._annotation_type = "POINTSET"

        return


##-

class PolyLine(AnnotationObject):
    """PolyLine: polygonal line (a sequence of segments)"""

    def __init__(self, coords, name=None, in_group="NO_GROUP"):
        """Initialize a POLYLINE object.

        Args:
            coords (list or tuple): coordinates of the points [(x0,y0), (x1,y1), ...]
                defining the segments (x0,y0)->(x1,y1); (x1,y1)->(x2,y2),...
            name (str): the name of the annotation
        """
        super().__init__(name, in_group)
        self._annotation_type = "POLYLINE"

        self.geom = shg.LineString(coords)

        return

    def duplicate(self):
        return PolyLine(self.xy, name=self.name, in_group=self._in_group)

    @staticmethod
    def empty_object():
        p = PolyLine([[0, 0], [1, 1]])
        return p

    @property
    def x(self) -> np.array:
        """Return the x coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        x, _ = self.geom.coords.xy
        return x

    @property
    def y(self) -> np.array:
        """Return the y coordinate(s) of the object. This is always a
        vector, even for a single point (when it has one element)."""
        _, y = self.geom.coords.xy
        return y

    @property
    def xy(self) -> np.array:
        """Return the xy-coordinates as a numpy.array [n,2]"""

        return np.array(self.geom.coords.xy).T

    @property
    def size(self) -> int:
        """Return the number of points defining the object."""
        return self.x.size

    def fromdict(self, d: dict) -> None:
        """Intialize the objct from a dictionary."""
        super().fromdict(d)
        self.geom = shg.LineString(zip(d["x"], d["y"]))

        return

    def asGeoJSON(self) -> dict:
        """Return a dictionary compatible with GeoJSON specifications."""
        return gj.Feature(geometry=gj.LineString(zip(self.x(), self.y())),
                          properties=dict(object_type="annotation",
                                          annotation_type=self._annotation_type,
                                          name=self._name)
                          )

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        if d["geometry"]["type"].lower() != "linestring":
            raise RuntimeError("Need a LineString feature! Got: " + str(d))

        super().fromGeoJSON(d)
        self._annotation_type = "POLYLINE"

        return


##-


##-
class Polygon(AnnotationObject):
    """Polygon: an ordered collection of points where the first and
    last points coincide."""

    def __init__(self, coords, name=None, in_group="NO_GROUP"):
        """Initialize a POLYGON annotation, i.e. a sequence of line
        segments forming a closed contour.

        Args:
            coords (list or tuple): coordinates of the points as in [(x0,y0), (x1,y1), ...]
            name (str): the name of the annotation
        """
        super().__init__(name, in_group)
        self._annotation_type = "POLYGON"

        self.geom = shg.Polygon(coords)

        return

    def duplicate(self):
        return Polygon(self.xy, name=self.name, in_group=self._in_group)

    @staticmethod
    def empty_object():
        p = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        return p

    def size(self) -> int:
        """Return the number of points defining the object."""
        return self.x.size

    @property
    def x(self) -> np.array:
        """Return the x coordinate(s) of the vertices of the EXTERIOR contour."""
        x, _ = self.geom.exterior.coords.xy
        return x

    @property
    def y(self) -> np.array:
        """Return the y coordinate(s) of the vertices of the EXTERIOR contour."""
        _, y = self.geom.exterior.coords.xy
        return y

    @property
    def xy(self) -> np.array:
        """Return the xy-coordinates of the vertices of the EXTERIOR contour,
        as a numpy.array [n,2]"""

        return np.array(self.geom.exterior.coords.xy).T

    def fromdict(self, d: dict) -> None:
        """Intialize the object from a dictionary."""
        super().fromdict(d)
        self.geom = shg.Polygon(zip(d["x"], d["y"]))

        return

    def fromGeoJSON(self, d: dict) -> None:
        """Initialize the object from a dictionary compatible with GeoJSON specifications."""
        if d["geometry"]["type"].lower() != "polygon":
            raise RuntimeError("Need a Polygon feature! Got: " + str(d))

        super().fromGeoJSON(d)
        self._annotation_type = "POLYGON"

        return
##-


##
## Import/export functions for interoperability with other formats.
##


def annotation_from_ASAP(
        infile: str,
        wsi_extent: tuple[int, int],
        magnification: float
) -> WSIAnnotation:
    """
    Import annotations from ASAP (XML) files. See also 
    https://github.com/computationalpathologygroup/ASAP
    """

    infile = Path(infile)
    with open(infile, 'r') as input:
        annot_dict = xmltodict.parse(input.read(), xml_attribs=True)

    if 'ASAP_Annotations' not in annot_dict:
        raise RuntimeError('Syntax error in ASAP XML file')

    annot_dict = annot_dict['ASAP_Annotations']
    if 'Annotations' not in annot_dict:
        raise RuntimeError('Syntax error in ASAP XML file')

    try:
        group_list = [g['@Name'] for g in annot_dict['AnnotationGroups']['Group']]
    except KeyError:
        group_list = []

    annot_list = annot_dict['Annotations']['Annotation']  # many nested levels...

    wsi_annotation = WSIAnnotation(infile.name, wsi_extent, magnification, group_list)
    for annot in annot_list:
        if annot['@Type'].lower() == 'dot':
            coords = [float(annot["Coordinates"]["Coordinate"]["@X"]), float(annot["Coordinates"]["Coordinate"]["@Y"])]
            obj = Point(coords, annot['@Name'], in_group=annot["@PartOfGroup"])
        elif annot['@Type'].lower() == 'pointset':
            coords = [(float(o["@X"]), float(o["@Y"])) for o in annot["Coordinates"]["Coordinate"]]
            obj = PointSet(coords, annot['@Name'], in_group=annot["@PartOfGroup"])
        elif annot['@Type'].lower() == 'polygon':
            coords = [(float(o["@X"]), float(o["@Y"])) for o in annot["Coordinates"]["Coordinate"]]
            obj = Polygon(coords, annot['@Name'], in_group=annot["@PartOfGroup"])
        else:
            raise RuntimeError(f"Unknown annotation type {annot['@Type']}")
        wsi_annotation.add_annotation_object(obj)

    return wsi_annotation


##
## Save annotations to Napari's CSV format.
##
def annotation_to_napari(annot: WSIAnnotation, csv_file: str, resolution_mpp: float = -1.0) -> None:
    """Save an <Annotation> in CSV files following Napari's specifications. Note that, since Points
        require a different format from other shapes, a separate file (with suffix '_points') will
        be created for storing all points in the annotation.

        :param annot: (Annotation) an annotation object
        :param asap_file: (str) name of the file for saving the annotation
        :param resolution_mpp: (float) if positive then the desired resolution (microns-per-pixel)
            for the saved annotation (all is scaled by accordingly).

        :return: True if everything is OK
        """

    # scale the annotation for the target:
    if resolution_mpp > 0.0:
        annot.set_mpp(resolution_mpp)

    points_file = Path(csv_file).with_name(Path(csv_file).stem + '_points.csv')
    shapes_file = Path(csv_file)

    points_lines = []
    points_idx = 0
    shapes_lines = []
    shapes_idx = 0

    for a in annot._annots:
        if a._annotation_type == "POINT":
            points_lines.append([points_idx, a.y(), a.x()])
            points_idx += 1
        elif a._annotation_type == "POLYGON":
            xy = a.xy()
            for k in np.arange(xy.shape[0]):
                shapes_lines.append([shapes_idx, 'polygon', k, xy[k, 1], xy[k, 0]])
            shapes_idx += 1
        elif a._annotation_type == "POINTSET":
            xy = a.xy()
            for k in np.arange(xy.shape[0]):
                shapes_lines.append([shapes_idx, 'path', k, xy[k, 1], xy[k, 0]])
            shapes_idx += 1
        else:
            raise RuntimeWarning("not implemented annotation type " + a._annotation_type)

    if points_idx > 0:
        # more than the header
        with open(points_file, 'w') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONE)
            writer.writerow(['index', 'axis-0', 'axis-1'])
            writer.writerows(points_lines)

    if shapes_idx > 0:
        # more than the header
        with open(shapes_file, 'w') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONE)
            writer.writerow(['index', 'shape-type', 'vertex-index', 'axis-0', 'axis-1'])
            writer.writerows(shapes_lines)

    return
