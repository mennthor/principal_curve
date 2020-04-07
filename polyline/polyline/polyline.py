import numpy as np


class Polyline(object):

    def __init__(self, vertices=None):
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

    def get_dist_to_line(self, pt):
        """
        Returns the smallest distance to any line segment in the polyline.

        Returns projection vectors from point to clostest point on the line, the
        distances and an array which encodes to which entity the distance was
        clostest to (integer >= 0: vertex id,
        integer < 0: -1 * (segment id + 1)).
        """
        # Call dist to segment for each segment, then filter results
        proj_vecs, dists, proj_to = [], [], []
        for seg_idx in range(len(self._segment_lengths)):
            _pv, _d, _pt = self._get_dist_to_segment(pt, seg_idx)
            proj_vecs.append(_pv)
            dists.append(_d)
            # Shift IDs to global indices, segments start at -1, -2, ..., -(n-1)
            m = _pt < 0
            _pt[m] = _pt[m] * (seg_idx + 1)
            _pt[~m] = _pt[~m] + seg_idx
            proj_to.append(_pt)

        # Argmin for all segments is closest to whole curve
        idx = np.argmin(dists, axis=0)
        proj_vecs = np.swapaxes(proj_vecs, 0, 1)
        dists = np.asarray(dists).T
        proj_to = np.asarray(proj_to).T
        proj_vecs = np.asarray([pvi[i] for pvi, i in zip(proj_vecs, idx)])
        dists = np.asarray([di[i] for di, i in zip(dists, idx)])
        proj_to = np.asarray([pti[i] for pti, i in zip(proj_to, idx)])

        return proj_vecs, dists, proj_to

    def get_angle_at_vertex(self, idx):
        """
        Returns the angle between segment (v_(i-1), v_i) and (v_i, v_(iü1) in
        radians. idx must be in [1, nvertices - 1].
        """
        try:
            idx + 1
        except Exception:
            raise ValueError(
                "Only a single vertex can be removed, not a range.")

        if idx < 1 or idx > self.nsegments:
            raise ValueError("idx must be in (1, nvertices)")

        v0 = self._vertices[idx - 1] - self._vertices[idx]
        v1 = self._vertices[idx] - self._vertices[idx + 1]
        v0 = v0 / np.linalg.norm(v0)
        v1 = v1 / np.linalg.norm(v1)
        return np.pi - np.arccos(np.dot(v0, v1))

    def append_vertex(self, v):
        """ Append vertices v to the end of the polyline """
        self.insert_vertex(v, self.nvertices)

    def insert_vertex(self, v, idx):
        """
        Insert vertices v to the polyline at position idx, before the existing
        element. If idx == nvertices, v is appended
        """
        try:
            idx + 1
        except Exception:
            raise ValueError(
                "Only a single vertex can be removed, not a range.")
        nvertices_before = self.nvertices
        if idx > nvertices_before or idx < 0:
            raise ValueError("index out of bounds for current polyline.")
        if len(v) == 0:
            raise ValueError("Invalid vertex given")

        v = np.atleast_2d(v)
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
        vertex idx - 1 and vertex idx (>0) if existing (polyline must have at
        least idx vertices).
        The distance to v0 and v1 is defined by scale, where scale = 0 means the
        new vertex is the same as v0, scale = 0.5 is exactly the middle between
        v0 and v1 and scale = 1 means the new vertex is the same as v1.
        """
        try:
            idx + 1
        except Exception:
            raise ValueError(
                "Only a single vertex can be removed, not a range.")

        if idx < 1 or idx > self.nsegments:
            raise ValueError("idx must be in (1, nvertices)")
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
        try:
            idx + 1
        except Exception:
            raise ValueError(
                "Only a single vertex can be removed, not a range.")

        self._vertices = np.delete(self._vertices, idx, axis=0)
        if self.nvertices < 2:
            self._clear_segment_lengths()
        else:
            if idx == 0:
                slen = self._get_segment_lengths(self._vertices[:2])
                self._segment_lengths = self._segment_lengths[1:]
            elif idx == self.nvertices - 1:
                slen = self._get_segment_lengths(self._vertices[-2:])
                self._segment_lengths = self._segment_lengths[:-1]
            else:
                slen = self._get_segment_lengths(
                    self._vertices[idx - 1:idx + 1])
                self._segment_lengths = np.concatenate((
                        self._segment_lengths[:idx - 1],
                        slen,
                        self._segment_lengths[idx:]))

    def replace_vertex(self, idx, v):
        """ Replace vertex at idx with v """
        try:
            idx + 1
        except Exception:
            raise ValueError(
                "Only a single vertex can be removed, not a range.")
        if idx >= self.nvertices or idx < 0:
            raise ValueError("index out of bounds for current polyline.")

        if len(v) == 0:
            raise ValueError("Invalid vertex given")

        self._vertices[[idx]] = np.atleast_2d(v)
        if self.nsegments > 1:
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

    def _get_dist_to_segment(self, pt, idx):
        """
        Get distance of pt to specified segment idx in the polyline.

        Returns projection vectors from point to clostest point on segment, the
        distances and an array which encodes to which entity the distance was
        clostest to (0: first vertex, -1: segment line, +1: second vertex).
        """
        pt = np.atleast_2d(pt)
        v0 = self._vertices[[idx]]
        v1 = self._vertices[[idx + 1]]
        # dproj = projection vector, dpt = pt vector relative to v0
        dproj = v1 - v0
        dpt = pt - v0
        norm_dproj = np.linalg.norm(dproj)
        norm_dpt = np.linalg.norm(dpt, axis=1)

        if np.isclose(norm_dproj, 0):
            # Both vertices are almost equal, return distance pt to v0
            return dpt, norm_dpt

        # Project dpt on line (v1, v0)
        dproj = dproj / norm_dproj
        proj_lens = np.dot(dpt, dproj.T)
        tangent_vecs = proj_lens * dproj

        # If projected length is not in [0, 1] it is outside the line element
        # and the distance to v0 (<0) or v1 (>len(pt)) is returned
        m0 = np.ravel(proj_lens < 0.)
        m1 = np.ravel(proj_lens > norm_dproj)
        # The projection vectors are orthogonal to the line or point to v0, v1
        proj_vecs = tangent_vecs - dpt  # projected on segment line
        proj_vecs[m0] = -dpt[m0]  # proj_len < 0
        proj_vecs[m1] = v1 - pt[m1]  # proj_len > norm_dproj

        # Encode if the distance is closest to vertex 0 (id=0), the segment
        # (id=-1) or vertex 1 (id=1)
        proj_to = np.zeros(len(proj_vecs), dtype=int) - 1
        proj_to[m0] = 0
        proj_to[m1] = 1

        return proj_vecs, np.linalg.norm(proj_vecs, axis=1), proj_to

    @property
    def nvertices(self):
        return len(self._vertices)

    @property
    def nsegments(self):
        return len(self._segment_lengths)

    @property
    def vertices(self):
        return self._vertices.copy()

    @property
    def segment_lengths(self):
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
