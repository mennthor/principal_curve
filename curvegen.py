import numpy as np


# Flaws: Not a real parametric curve, densitiy of sampled points change
#        around line when derivative get larger
class Curve2D(object):

    def __init__(self, curvature=0, curve_rel_start=0):
        """
        curvature = scaling factor of first quarter of sine wave
        curve_rel_start in [0, 1] -> 0 = curve start at beginning, 1 = at end
        """
        if curve_rel_start < 0 or curve_rel_start > 1:
            raise ValueError("curve_rel_start must be in [0, 1]")
        self.curvature = curvature
        self.curve_rel_start = curve_rel_start

    def sample(self, x, stddev=1., seed=None):
        """
        Model: (x, y) = (x, f(x)) + rot(alpha) * norm(0, stddev)
        where rot(alpha) rotates the point to the normal vector of the curve at
        f(x) (samples with stddev along the normal vector at each (x, f(x))).
        """
        x = np.asarray(x)
        y, dy = self(x)
        # Sample offset along normal vector
        rndgen = np.random.RandomState(seed)
        # y_sam = rndgen.normal(0, stddev, size=x.shape)
        y_sam = rndgen.uniform(-stddev, stddev, size=x.shape)
        y_sam = rndgen.normal(0, stddev, size=x.shape)
        x_sam = np.zeros_like(x)
        # Rotate sampled points along normal vector
        dx = np.ones_like(x)
        angles = np.arctan2(dy, dx)
        s, c = np.sin(angles), np.cos(angles)
        rot = np.asarray([[[ci, -si], [si, ci]] for si, ci in zip(s, c)])
        vec = np.asarray([
            [xi, yi] for xi, yi in zip(x_sam, y_sam)])
        rot_vec = np.asarray([
            np.matmul(roti, veci) for roti, veci in zip(rot, vec)])

        # print("x, y", x, y)
        # print("dx, dy", dx, dy)
        # print("angles", np.rad2deg(angles))
        # print("vecs", vec)
        # print("rotmats", rot)
        # print("rot vecs", rot_vec)

        return x + rot_vec[:, 0], y + rot_vec[:, 1]

    def __call__(self, x):
        """
        Get y values of the curve for given x
        """
        x = np.asarray(x)
        # Quarter sine period for the curvature starting at curve_rel_start
        xmax = np.amax(x)
        xmin = self.curve_rel_start * xmax
        normed = np.zeros_like(x)
        m = x > xmax * self.curve_rel_start
        # Transform to [pi / 2, pi]
        normed[m] = (x[m] - xmin) / (xmax - xmin) * np.pi / 2. + np.pi / 2.
        shift = np.zeros_like(x)
        shift[m] = 1.
        func = self.curvature * (np.sin(normed) - shift)
        # Get gradient (note: respect inner derivative from normalization)
        grad = np.zeros_like(x)
        grad[m] = (self.curvature * np.cos(normed[m]) /
                   (xmax - xmin) * np.pi / 2.)
        return func, grad
