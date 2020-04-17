import numpy as np


class Polyline(object):

    def __init__(self, vertices=None):
        """ Vertices must have shape (nvertices, 2). """
        self._vertices = np.empty((0, 2))
        self._segment_lengths = np.empty(0)
        if vertices is not None:
            vertices = np.atleast_2d(vertices)
            if len(vertices.shape) != 2 or vertices.shape[1] != 2:
                return
            for v in vertices:
                self.append_vertex(v)

    def get_value(self, t):
        """
        Return the (interpolated) value at the given parameter t (in [0, 1])
        """
        t = np.clip(np.asarray(t), 0., 1.)
        # Select segments to interpolate each t in
        cumlen = (np.cumsum(self._segment_lengths) /
                  np.sum(self._segment_lengths))
        cumlen = np.insert(cumlen, 0, 0.)
        segment_idx = np.searchsorted(cumlen, t)
        # Handle first segment separately due to index out of bounds
        m = segment_idx > 0
        v0 = np.repeat(self._vertices[[0]], repeats=len(t), axis=0)
        v1 = np.repeat(self._vertices[[1]], repeats=len(t), axis=0)
        v0[m] = self._vertices[segment_idx - 1][m]
        v1[m] = self._vertices[segment_idx][m]

        # Normalize ts in [0, 1] relative to corresponding segment
        segment_idx = np.maximum(0, segment_idx - 1)
        t_norm = ((t - cumlen[segment_idx]) /
                  (cumlen[segment_idx + 1] - cumlen[segment_idx]))

        # Interpolated x and y values at ts
        v = (v1 - v0) * t_norm.reshape(len(t), 1) + v0
        return v

    def get_dist_to_line(self, pts):
        """
        Returns the smallest distance to any line segment in the polyline.
        Points must have shape (npoints, 2).

        Returns projection vectors from point to clostest point on the line, the
        distances and an array which encodes to which entity the distance was
        clostest to (integer >= 0: vertex id,
        integer < 0: -1 * (segment id + 1)).

        The returned gradients are with respect to varying each given data point
        and only valid for the current projection entity (on segment or vertex).
        Gradients are not continuous at projected segment borders and thus only
        valid if the corresponding point does not project onto another one.

        Returns: dists, dists_grad, proj_to, proj_vecs
        """
        pts = np.atleast_2d(pts)
        if len(pts) == 0 or len(pts.shape) != 2 or pts.shape[1] != 2:
            raise ValueError("pts must have shape (npts, 2).")

        # Call dist to segment for each segment, then filter results
        dists, dists_grad, proj_to, proj_vecs = [], [], [], []
        for seg_idx in range(self.nsegments):
            _d, _dg, _pt, _pv = self._get_dist_to_segment(pts, seg_idx)
            dists.append(_d)  # shape (nsegments, npts)
            dists_grad.append(_dg)  # shape (nsegments, npts, 2, 2{x,y})
            proj_vecs.append(_pv)  # shape (nsegments, npts, 2{x,y})
            # Shift local segment projection IDs to global indices
            m = _pt < 0
            _pt[m] = _pt[m] * (seg_idx + 1)  # segments: -1, -2, ..., -nsegs
            _pt[~m] = _pt[~m] + seg_idx  # Vertices: 0, 1, ..., nverts - 1
            proj_to.append(_pt)  # shape (nsegments, npts)

        # Argmin over all segments is closest distance from point to whole curve
        # Closest segment idx for each point, shape (npts)
        idx = np.argmin(dists, axis=0)
        # Bring in correct shape to broadcast argmin indices for all points
        dists = np.asarray(dists).T  # shape (npts, nsegments)
        dists_grad = np.swapaxes(dists_grad, 0, 1)  # (npts, nsegs, 2, 2{x,y})
        proj_to = np.asarray(proj_to).T  # shape (npts, nsegments)
        proj_vecs = np.swapaxes(proj_vecs, 0, 1)  # shape (npts, nsegs, 2{x,y})
        # Put to output, avoids hefty quadtratic numpy broadcasting
        dists = np.asarray([di[i] for i, di in zip(idx, dists)])
        dists_grad = np.asanyarray([dgi[i] for i, dgi in zip(idx, dists_grad)])
        proj_to = np.asarray([pti[i] for i, pti in zip(idx, proj_to)])
        proj_vecs = np.asarray([pvi[i] for i, pvi in zip(idx, proj_vecs)])

        return dists, dists_grad, proj_to, proj_vecs

    def get_cos_angle_at_vertex(self, idx):
        """
        Returns cos(angle) between segment (v_(i-1), v_i) and (v_i, v_(i+1).
        idx must be in [1, nvertices - 1] and the gradient in (x, y) direction
        for the given vertex coordinates.
        """
        self._check_idx(idx)

        if idx < 1 or idx > self.nsegments:
            raise ValueError("idx must be in [1, nsegments]")

        v0 = -self._vertices[idx - 1] + self._vertices[idx]  # Segment (i-1, i)
        v1 = -self._vertices[idx] + self._vertices[idx + 1]  # Segment (i, i+1)
        normv0, normv1 = np.linalg.norm(v0), np.linalg.norm(v1)
        if np.isclose(normv1, 0.) or np.isclose(normv1, 0.):
            return 0, 0

        cos_angle = np.dot(v0, v1) / normv0 / normv1

        # Precomputed analytic gradient: [d/dx cos_angle, d/dy cos_angle]
        grad_x = ((v1[0] - v0[0]) / normv0 / normv1 +
                  cos_angle * (v1[0] / normv1**2 - v0[0] / normv0**2))
        grad_y = ((v1[1] - v0[1]) / normv0 / normv1 +
                  cos_angle * (v1[1] / normv1**2 - v0[1] / normv0**2))

        # pi - ang because straight line with 2 vertices has angle 180°
        # -> cos(pi - ang) = cos(pi - arccos(x)) = -cos(arccos(x)) = -x
        return -cos_angle, np.array([-grad_x, -grad_y])

    def get_angle_at_vertex(self, idx):
        """
        Returns the angle between segment (v_(i-1), v_i) and (v_i, v_(i+1) in
        radians. idx must be in [1, nvertices - 1].
        """
        cos_angle, cos_angle_grad = self.get_cos_angle_at_vertex(idx)
        # Apply chain rule to arccos(cos_angle) gradient
        angle = np.arccos(cos_angle)
        # d/dx arccos(cos_angle(x)) = -1/sqrt(1-cos_angle) * d/dx(cos_angle)
        angle_grad = -1. / (np.sqrt(1. - cos_angle)) * cos_angle_grad
        return angle, angle_grad

    def append_vertex(self, v):
        """ Append vertices v to the end of the polyline """
        self.insert_vertex(v, self.nvertices)

    def insert_vertex(self, v, idx):
        """
        Insert vertices v to the polyline at position idx, before the existing
        element. If idx == nvertices, v is appended
        """
        self._check_idx(idx)

        nvertices_before = self.nvertices
        if idx < 0 or idx > nvertices_before:
            raise ValueError("Index out of bounds for current polyline.")

        v = np.atleast_2d(v)
        if len(v) == 0 or len(v.shape) != 2 or v.shape[1] != 2:
            raise ValueError("Single vertex must have shape (2,) or (1, 2)")

        ninserted = v.shape[0]
        self._vertices = np.insert(self._vertices, idx, v, axis=0)
        if self.nvertices > 1:
            # Collect connections that changed for segment lengths recalculation
            imin = max(0, idx - 1)
            imax = min(self.nvertices, idx + ninserted + 1)
            lens = self._get_segment_lengths(self._vertices[imin:imax])
            if idx == 0:
                self._segment_lengths = np.concatenate(
                    (lens, self._segment_lengths))
            elif idx == nvertices_before:
                self._segment_lengths = np.concatenate(
                    (self._segment_lengths, lens))
            else:
                self._segment_lengths = np.concatenate((
                    self._segment_lengths[:idx - 1],
                    lens,
                    self._segment_lengths[idx:]))

    def insert_vertex_between(self, idx, scale):
        """
        Inserts a new vertex on the segment before vertex at idx, so between
        vertex idx-1 and vertex idx (>0) if existing (polyline must have at
        least idx vertices).
        The distance to v0 and v1 is defined by scale, where scale = 0 means the
        new vertex is the same as v0, scale = 0.5 is exactly the middle between
        v0 and v1 and scale = 1 means the new vertex is the same as v1.
        """
        self._check_idx(idx)

        if idx < 1 or idx > self.nsegments:
            raise ValueError("idx must be in [1, nsegments]")

        if scale < 0. or scale > 1.:
            raise ValueError("scale must be in [0, 1]")

        # If scale is 0 or 1, just duplicate the existing vertices
        if scale == 0.:
            new_v = self.vertices[idx - 1]
        elif scale == 1.:
            new_v = self.vertices[idx]
        else:
            v0, v1 = self.vertices[idx - 1], self.vertices[idx]
            new_v = scale * (v1 - v0) + v0

        self.insert_vertex(new_v, idx)  # Insert before v1 (between v0, v1)

    def remove_vertex(self, idx):
        """ Removes vertex at idx (in [0, nvertices-1]) """
        self._check_idx(idx)

        if idx < 0 or idx > self.nvertices - 1:
            raise ValueError("idx must be in [0, nvertices - 1]")

        nvertices_before = self.nvertices
        self._vertices = np.delete(self._vertices, idx, axis=0)
        if self.nvertices < 2:
            self._clear_segment_lengths()
        else:
            if idx == 0:
                self._segment_lengths = self._segment_lengths[1:]
            elif idx == nvertices_before - 1:
                self._segment_lengths = self._segment_lengths[:-1]
            else:
                slen = self._get_segment_lengths(
                    self._vertices[idx - 1:idx + 1])
                self._segment_lengths = np.concatenate((
                        self._segment_lengths[:idx - 1],
                        slen,
                        self._segment_lengths[idx + 1:]))

    def replace_vertex(self, idx, v):
        """ Replace vertex at idx with v """
        self._check_idx(idx)

        if idx < 0 or idx > self.nvertices - 1:
            raise ValueError("Index out of bounds for current polyline.")

        v = np.atleast_2d(v)
        if len(v) == 0 or len(v.shape) != 2 or v.shape[1] != 2:
            raise ValueError("Single vertex must have shape (2,) or (1, 2)")

        self._vertices[[idx]] = v
        if self.nsegments > 0:
            if idx == 0:
                slen = self._get_segment_lengths(self._vertices[:2])
                self._segment_lengths[0] = slen
            elif idx == self.nvertices - 1:
                slen = self._get_segment_lengths(self._vertices[-2:])
                self._segment_lengths[-1] = slen
            else:
                # Update 2 segments
                slen = self._get_segment_lengths(
                    self._vertices[idx - 1:idx + 2])
                self._segment_lengths[idx - 1:idx + 1] = slen

    def translate(self, trans_vec):
        """ Translates the whole line by shifting each vertex by trans_vec """
        self._vertices = self._vertices + np.atleast_2d(trans_vec).reshape(1, 2)

    def get_vertex(self, idx):
        return self._vertices[idx]

    def get_segment_length(self, idx):
        return self._segment_lengths[idx]

    def get_segment_length_grad(self, idx, which="last"):
        """
        Returns the gradient [dL/dx, dL/dy] for the segment idx from vertex
        v_idx to v_(idx+1) with respect to moving one of the vertices.
        If 'which' is 'first', then the gradient for the first vertex is
        calcualated, if 'last', then the second vertex is used for the gradient.
        """
        self._check_idx(idx)

        if which not in ["first", "last"]:
            raise ValueError("'which' must be one of 'first', 'last'.")

        # L = sqrt((v1x - v0x)**2 + (v1y - v0y)**2)  for segment (v0->v1)
        # dL/dx = +(v1x - v0x) / L  for x with respect to v1
        # dL/dx = -(v1x - v0x) / L  for x with respect to v0
        L = self.get_segment_length(idx)
        v0, v1 = self._vertices[idx:idx + 2]
        dv = v1 - v0
        if which == "first":
            dv = -dv  # Gradient changes sign if first vertex is moved
        return dv / L

    def clear(self):
        self._clear_vertices()
        self._clear_segment_lengths()

    def _clear_vertices(self):
        self._vertices = np.empty((0, 2))

    def _clear_segment_lengths(self):
        self._segment_lengths = np.empty(0)

    def _get_segment_lengths(self, v):
        if len(v) < 2:
            return np.empty(0)
        return np.asarray(np.sqrt(
            (v[:-1, 0] - v[1:, 0])**2 + (v[:-1, 1] - v[1:, 1])**2))

    def _get_dist_to_segment(self, pts, idx):
        """
        Get distance of pts to specified segment idx in the polyline.
        idx is integer, pts has shape (npts, 2).

        Returns projection vectors from point to clostest point on segment, the
        distances and an array which encodes to which entity the distance was
        clostest to (0: first vertex, -1: segment line, +1: second vertex).

        The returned gradients are with respect to varying each vertex of the
        segment for all given points and only valid for the current projection
        entity (on segment or vertex) and not continous at the region borders.
        For each point, two gradients are returned, with respect to variation of
        the first and second vertex of the segment respectively.
        If a point projects onto one of the vertices, the second gradient is set
        to zero, because the other vertex doesn't affect the distance at all and
        the association from projection ID to gradient must be unambiguous.
        Gradients for points with distance 0 are both set to zero.

        Returns: dists, dists_grad, proj_to, proj_vecs
        """
        v0 = self._vertices[[idx]]  # constant, shape (1, 2)
        v1 = self._vertices[[idx + 1]]  # constant, shape (1, 2)
        # dproj = projection vector, dpt_v0 = pts vector relative to v0
        dproj = v1 - v0  # constant, shape (1, 2)
        dpt_v0 = pts - v0  # shape (npts, 2)
        norm_dproj = np.linalg.norm(dproj, axis=1)
        norm_dpt_v0 = np.linalg.norm(dpt_v0, axis=1).reshape(len(dpt_v0), 1)

        if np.isclose(norm_dproj, 0):
            # Both vertices are almost equal, return distance pts to v0. All
            # points project to v0 so gradients are the normalized projection
            # vectors, which are simply the connection vectors (pt -> v0) here
            dists_grad = np.zeros((len(pts), 2, 2), dtype=np.float)
            dists_grad[:, :, 0] = -dpt_v0 / norm_dpt_v0
            return (norm_dpt_v0, dists_grad,
                    np.zeros(len(dpt_v0), dtype=int), dpt_v0)

        # Project dpt_v0 on line (v0, v1)
        dproj = dproj / norm_dproj
        proj_lens = np.dot(dpt_v0, dproj.T)
        tangent_vecs = proj_lens * dproj

        # If projected length is not in [0, norm_dproj] it is outside the line
        # element and the distance to v0 (<0) or v1 (>norm_dproj) is returned
        m0 = np.squeeze(proj_lens < 0., axis=1)
        m1 = np.squeeze(proj_lens > norm_dproj, axis=1)
        # The projection vectors are orthogonal to the line or point to v0, v1
        proj_vecs = tangent_vecs - dpt_v0  # projected on segment line
        proj_vecs[m0] = -dpt_v0[m0]  # proj_len < 0 -> project to v0
        proj_vecs[m1] = v1 - pts[m1]  # proj_len > norm_dproj -> project to v1

        # Encode if the distance is closest to vertex 0 (id=0), the segment
        # (id=-1) or vertex 1 (id=1)
        proj_to = np.zeros(len(proj_vecs), dtype=int) - 1
        proj_to[m0] = 0
        proj_to[m1] = 1

        # Get distances
        dists = np.linalg.norm(proj_vecs, axis=1)

        # Get distance gradients [dL/dx, dL/dy]
        # shape (npts, ngradients per point, (x, y)) -> (npts, 2, 2)
        dists_grad = np.empty((len(dists), 2, 2), dtype=np.float)
        valid = dists > 0
        # Points projecting on v0
        _m = np.logical_and(valid, m0)
        if np.any(_m):
            _grad = np.zeros((np.sum(_m), 2, 2))
            _grad[:, 0] = -dpt_v0[_m] / norm_dpt_v0[_m]
            dists_grad[_m] = _grad

        # Points projecting on v1
        _m = np.logical_and(valid, m1)
        if np.any(_m):
            dpt_v1 = v1 - pts[_m]
            _grad = np.zeros((np.sum(_m), 2, 2))
            _grad[:, 0] = (
                dpt_v1 / np.linalg.norm(dpt_v1, axis=1).reshape(len(dpt_v1), 1))
            dists_grad[_m] = _grad

        # Points projecting on segment (v0, v1)
        mseg = np.logical_and(~m0, ~m1)
        _m = np.logical_and(valid, mseg)
        if np.any(_m):
            _grad = np.empty((np.sum(_m), 2, 2))
            _grad[:, 0] = (
                1. - dproj**2 -
                dpt_v0[_m] / norm_dproj * dproj +
                proj_lens[_m] * 2 * dproj**2 / norm_dproj -
                proj_lens[_m] / norm_dproj)
            _grad[:, 1] = (
                dpt_v0[_m] / norm_dproj * dproj -
                proj_lens[_m] * 2 * dproj**2 / norm_dproj +
                proj_lens[_m] / norm_dproj)
            dists_grad[_m] = _grad

        return dists, dists_grad, proj_to, proj_vecs

    def _check_idx(self, idx):
        """ Checks if idx is a single integer, otherwise logic breaks """
        err = "idx must be an integer."
        try:
            idx + 1
        except Exception:
            raise ValueError(err)
        if isinstance(idx, np.ndarray):
            raise ValueError(err)

    def _check_vertices(self, v):
        return

    @property
    def nvertices(self):
        return len(self._vertices)

    @property
    def nsegments(self):
        return len(self._segment_lengths)

    @property
    def vertices(self):
        """ Returns a copy of the internal vertex array """
        return self._vertices.copy()

    @property
    def segment_lengths(self):
        """ Returns a copy of the internal segment length array """
        return self._segment_lengths.copy()

    @property
    def line_length(self):
        return np.sum(self._segment_lengths)

    @property
    def xcoords(self):
        return self._vertices[:, 0]

    @property
    def ycoords(self):
        return self._vertices[:, 1]
