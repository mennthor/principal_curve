#!/usr/bin/env python
# coding: utf-8

"""
Demonstrates a rotation of a whole axis object by 90° counter clockwise, which
is effectlivly a transformation x->y, y->-x.
This can be used, when ego car coordinates should be plotted, which have by
convention the x-axis to the front in driving direction.

The advantage of the axis rotation is, that no data transformation must be used.
Plot the x, y coordinates and set limits and labels as usual, the transformation
then rotates the whole axis object around properly.

Note: Not tested with all possible plots.
"""

from matplotlib import transforms
from matplotlib.collections import Collection, PatchCollection, PathCollection
from matplotlib.patches import Patch, PathPatch
from matplotlib.lines import Line2D
from matplotlib.image import AxesImage


def rot_axis_90deg_cc(ax):
    """
    Rotates a whole axis by changing the rotation elements and adjusting limits
    and labels, so that x is the up axis, and y is to the left (righthanded).
    Use x, y coordinates and labels as intended and call this method at last
    step before showing the figure.

    From: stackoverflow.com/questions/43892973
    """
    # Add transformation to axis elements.
    # Order matters: first rotate, then base
    # Add 90° rotation to achieve desired trafo (x->y, y->-x)
    rot = transforms.Affine2D().rotate_deg(90)
    # Why does this not work?
    # consider = (Collection, PatchCollection, PathCollection,
    #             Patch, PathPatch, Line2D, AxesImage)
    # for child in ax.get_children():
    #   if not isinstance(child, consider):
    #        continue
    for child in ax.images + ax.lines + ax.collections:
        trans = child.get_transform()
        child.set_transform(rot + trans)
        if isinstance(child, PathCollection):
            transoff = child.get_offset_transform()
            child._transOffset = rot + transoff

    # "Rotate" axis limits and labels
    _lim = ax.axis()
    ax.axis(_lim[2:4][::-1] + _lim[0:2])

    # "Rotate" labels
    _xlab = ax.xaxis.get_label().get_text()
    ax.set_xlabel(ax.yaxis.get_label().get_text())
    ax.set_ylabel(_xlab)

    return ax
