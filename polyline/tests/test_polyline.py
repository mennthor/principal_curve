from polyline import Polyline

from pytest import approx, raises
from math import sqrt, pi


def norm(x0, y0, x1, y1):
    return sqrt((x0 - x1)**2 + (y0 - y1)**2)


def test_norm_helper():
    assert 0 == approx(norm(2, 3, 2, 3))
    assert 1 == approx(norm(2, 3, 2, 4))
    assert 1 == approx(norm(-2, 3, -2, 4))
    assert 2 == approx(norm(0, -3, 0, -1))
    assert 2 == approx(norm(2, -1, 0, -1))
    assert sqrt(2) == approx(norm(1, 1, 2, 2))


# #############################################################################
# <Test polyline: Constructor>
class TestPolylineConstructor():

    def test_constructor_empty(self):
        v = []
        pl = Polyline(v)

        assert pl.nvertices == len(pl.vertices) == 0
        assert len(pl.segment_lengths) == pl.nsegments == 0

    def test_constructor_one_vertex(self):
        v = [[1, 2]]
        pl = Polyline(v)

        assert pl.nvertices == len(pl.vertices) == 1
        assert len(pl.segment_lengths) == pl.nsegments == 0

        assert pl.vertices[0, 0] == v[0][0]
        assert pl.vertices[0, 1] == v[0][1]

    def test_constructor_two_vertices(self):
        v = [[1, 2], [3, 4]]
        pl = Polyline(v)

        assert pl.nvertices == len(pl.vertices) == 2
        assert len(pl.segment_lengths) == pl.nsegments == 1

        for i, vi in enumerate(v):
            assert pl.vertices[i, 0] == vi[0]
            assert pl.vertices[i, 1] == vi[1]

        assert pl.segment_lengths[0] == approx(
            norm(v[0][0], v[0][1], v[1][0], v[1][1]))

    def test_constructor_many_vertices(self):
        v = [[1, 2], [3, 4], [4, 5], [7, 8], [9, 10]]
        pl = Polyline(v)

        assert pl.nvertices == len(pl.vertices) == len(v)
        assert len(pl.segment_lengths) == pl.nsegments == len(v) - 1

        for i, vi in enumerate(v):
            assert pl.vertices[i, 0] == vi[0]
            assert pl.vertices[i, 1] == vi[1]

            if i < len(v) - 1:
                assert pl.segment_lengths[i] == approx(
                    norm(vi[0], vi[1], v[i + 1][0], v[i + 1][1]))
# </Test polyline: Constructor>
# #############################################################################


# #############################################################################
# <Test polyline: Append vertices>
class TestPolylineAppendVertices():

    def test_append_empty(self):
        pl = Polyline()
        v = []
        with raises(ValueError):
            pl.append_vertex(v)

    def test_append_one_vertex(self):
        pl = Polyline()
        v = [[1, 2]]
        pl.append_vertex(v)

        assert pl.nvertices == len(pl.vertices) == 1
        assert len(pl.segment_lengths) == pl.nsegments == 0

        assert pl.vertices[0, 0] == v[0][0]
        assert pl.vertices[0, 1] == v[0][1]

    def test_append_two_vertices(self):
        pl = Polyline()
        v = [[1, 2], [3, 4]]
        pl.append_vertex(v)

        assert pl.nvertices == len(pl.vertices) == 2
        assert len(pl.segment_lengths) == pl.nsegments == 1

        for i, vi in enumerate(v):
            assert pl.vertices[i, 0] == vi[0]
            assert pl.vertices[i, 1] == vi[1]

        assert pl.segment_lengths[0] == approx(
            norm(v[0][0], v[0][1], v[1][0], v[1][1]))

    def test_append_many_vertices(self):
        pl = Polyline()
        v = [[1, 2], [3, 4], [4, 5], [7, 8], [9, 10]]
        pl.append_vertex(v)

        assert pl.nvertices == len(pl.vertices) == len(v)
        assert len(pl.segment_lengths) == pl.nsegments == len(v) - 1

        for i, vi in enumerate(v):
            assert pl.vertices[i, 0] == vi[0]
            assert pl.vertices[i, 1] == vi[1]

            if i < len(v) - 1:
                assert pl.segment_lengths[i] == approx(
                    norm(vi[0], vi[1], v[i + 1][0], v[i + 1][1]))
# </Test polyline: Append vertices>
# #############################################################################


# #############################################################################
# <Test polyline: Insert vertices>
class TestPolylineInsertVertices():

    def test_insert_single_shape(self):
        pl = Polyline()
        v = [1, 2]  # Single vertex in non-2D form should also be accepted
        pl.insert_vertex(v, 0)

        assert pl.nvertices == len(pl.vertices) == 1
        assert len(pl.segment_lengths) == pl.nsegments == 0

        assert pl.vertices[0, 0] == v[0]
        assert pl.vertices[0, 1] == v[1]

    def test_insert_in_empty(self):
        pl = Polyline()
        v = [[1, 2]]
        with raises(ValueError):
            pl.insert_vertex(v, 1)  # 1 is out of bounds for empty polyline
        with raises(ValueError):
            pl.insert_vertex(v, -1)  # -1 is out of bounds

        pl.insert_vertex(v, 0)  # Proper insert

        assert pl.nvertices == len(pl.vertices) == 1
        assert len(pl.segment_lengths) == pl.nsegments == 0

        assert pl.vertices[0, 0] == v[0][0]
        assert pl.vertices[0, 1] == v[0][1]

    def test_insert_in_len_one_polyline(self):
        v = [[1, 2]]
        pl = Polyline(v)

        v_ins = [[3, 4]]
        pl.insert_vertex(v_ins, 0)  # Insert at beginning
        pl.insert_vertex(v_ins, 2)  # Insert at end

        assert pl.nvertices == len(pl.vertices) == 3
        assert len(pl.segment_lengths) == pl.nsegments == 2

        for i in [0, 2]:
            assert pl.vertices[i, 0] == v_ins[0][0]
            assert pl.vertices[i, 1] == v_ins[0][1]
        assert pl.vertices[1, 0] == v[0][0]
        assert pl.vertices[1, 1] == v[0][1]

    def test_insert_multiple_random(self):
        v = [[1, 2], [3, 4], [5, 6], [7, 8]]
        pl = Polyline(v)

        v_ins = [[9, 10], [11, 12], [13, 14]]
        pl.insert_vertex(v_ins, 2)
        pl.insert_vertex(v_ins, 5)

        v_expected = v[:2] + v_ins + v[2:]
        v_expected = v_expected[:5] + v_ins + v_expected[5:]

        assert pl.nvertices == len(pl.vertices) == len(v_expected)
        assert len(pl.segment_lengths) == pl.nsegments == pl.nvertices - 1

        for i, vi in enumerate(v_expected):
            assert pl.vertices[i, 0] == vi[0]
            assert pl.vertices[i, 1] == vi[1]
            if i < pl.nsegments - 1:
                assert pl.segment_lengths[i] == approx(norm(
                    vi[0], vi[1], v_expected[i + 1][0], v_expected[i + 1][1]))

    def test_insert_between(self):
        v = [[1, 0], [1, 1]]
        pl = Polyline(v)

        pl.insert_vertex_between(1, scale=0.25)
        v_expected = v[:1] + [[1, 0.25]] + v[1:]

        assert pl.nvertices == len(pl.vertices) == len(v_expected)
        assert len(pl.segment_lengths) == pl.nsegments == pl.nvertices - 1

        for i, vi in enumerate(v_expected):
            assert pl.vertices[i, 0] == approx(vi[0])
            assert pl.vertices[i, 1] == approx(vi[1])
            if i < pl.nsegments - 1:
                assert pl.segment_lengths[i] == approx(norm(
                    vi[0], vi[1], v_expected[i + 1][0], v_expected[i + 1][1]))

    def test_insert_between_scale_0(self):
        v = [[1, 0], [1, 1]]
        pl = Polyline(v)

        pl.insert_vertex_between(1, scale=0)
        v_expected = v[:1] + v

        assert pl.nvertices == len(pl.vertices) == len(v_expected)
        assert len(pl.segment_lengths) == pl.nsegments == pl.nvertices - 1

        for i, vi in enumerate(v_expected):
            assert pl.vertices[i, 0] == vi[0]
            assert pl.vertices[i, 1] == vi[1]
            if i < pl.nsegments - 1:
                assert pl.segment_lengths[i] == approx(norm(
                    vi[0], vi[1], v_expected[i + 1][0], v_expected[i + 1][1]))

    def test_insert_between_scale_1(self):
        v = [[1, 0], [1, 1]]
        pl = Polyline(v)

        pl.insert_vertex_between(1, scale=1)
        v_expected = v + v[1:]

        assert pl.nvertices == len(pl.vertices) == len(v_expected)
        assert len(pl.segment_lengths) == pl.nsegments == pl.nvertices - 1

        for i, vi in enumerate(v_expected):
            assert pl.vertices[i, 0] == vi[0]
            assert pl.vertices[i, 1] == vi[1]
            if i < pl.nsegments - 1:
                assert pl.segment_lengths[i] == approx(norm(
                    vi[0], vi[1], v_expected[i + 1][0], v_expected[i + 1][1]))
# </Test polyline: Insert vertices>
# #############################################################################


# #############################################################################
# <Test polyline: Remove vertex>
class TestPolylineRemoveVertex():
    def test_remove_vertex_beginning(self):
        v = [[1, 3], [3, 5], [6, 2], [2, 2]]
        pl = Polyline(v)
        pl.remove_vertex(0)

        v_expected = v[1:]
        assert len(pl.vertices) == pl.nvertices == len(v_expected)
        assert len(pl.segment_lengths) == pl.nsegments == pl.nvertices - 1

        for i, vi in enumerate(v_expected):
            assert pl.vertices[i, 0] == vi[0]
            assert pl.vertices[i, 1] == vi[1]
            if i < pl.nsegments - 1:
                assert pl.segment_lengths[i] == approx(norm(
                    vi[0], vi[1], v_expected[i + 1][0], v_expected[i + 1][1]))

    def test_remove_vertex_end(self):
        v = [[1, 3], [3, 5], [6, 2], [2, 2]]
        pl = Polyline(v)
        pl.remove_vertex(len(v) - 1)

        v_expected = v[:-1]
        assert len(pl.vertices) == pl.nvertices == len(v_expected)
        assert len(pl.segment_lengths) == pl.nsegments == pl.nvertices - 1

        for i, vi in enumerate(v_expected):
            assert pl.vertices[i, 0] == vi[0]
            assert pl.vertices[i, 1] == vi[1]
            if i < pl.nsegments - 1:
                assert pl.segment_lengths[i] == approx(norm(
                    vi[0], vi[1], v_expected[i + 1][0], v_expected[i + 1][1]))

    def test_remove_vertex_middle(self):
        v = [[1, 3], [3, 5], [6, 2], [2, 2]]
        pl = Polyline(v)
        # Order matters
        pl.remove_vertex(2)
        pl.remove_vertex(1)

        v_expected = v[:1] + v[3:]
        assert len(pl.vertices) == pl.nvertices == len(v_expected)
        assert len(pl.segment_lengths) == pl.nsegments == pl.nvertices - 1

        for i, vi in enumerate(v_expected):
            assert pl.vertices[i, 0] == vi[0]
            assert pl.vertices[i, 1] == vi[1]
            if i < pl.nsegments - 1:
                assert pl.segment_lengths[i] == approx(norm(
                    vi[0], vi[1], v_expected[i + 1][0], v_expected[i + 1][1]))
# </Test polyline: Remove vertex>
# #############################################################################


# #############################################################################
# <Test polyline: Replace vertex>
class TestPolylineReplaceVertex():
    def test_replace_vertex_beginning(self):
        v = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        pl = Polyline(v)
        v_repl = [-1, -1]
        pl.replace_vertex(0, v_repl)

        v_expected = [v_repl] + v[1:]
        assert len(pl.vertices) == pl.nvertices == len(v_expected)
        assert len(pl.segment_lengths) == pl.nsegments == pl.nvertices - 1

        for i, vi in enumerate(v_expected):
            assert pl.vertices[i, 0] == vi[0]
            assert pl.vertices[i, 1] == vi[1]
            if i < pl.nsegments - 1:
                assert pl.segment_lengths[i] == approx(norm(
                    vi[0], vi[1], v_expected[i + 1][0], v_expected[i + 1][1]))

    def test_replace_vertex_end(self):
        v = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        pl = Polyline(v)
        v_repl = [-1, -1]
        pl.replace_vertex(len(v) - 1, v_repl)

        v_expected = v[:-1] + [v_repl]
        assert len(pl.vertices) == pl.nvertices == len(v_expected)
        assert len(pl.segment_lengths) == pl.nsegments == pl.nvertices - 1

        for i, vi in enumerate(v_expected):
            assert pl.vertices[i, 0] == vi[0]
            assert pl.vertices[i, 1] == vi[1]
            if i < pl.nsegments - 1:
                assert pl.segment_lengths[i] == approx(norm(
                    vi[0], vi[1], v_expected[i + 1][0], v_expected[i + 1][1]))

    def test_replace_vertex_middle(self):
        v = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        pl = Polyline(v)
        v_repl = [-1, -1]
        pl.replace_vertex(2, v_repl)

        v_expected = v[:2] + [v_repl] + v[3:]
        assert len(pl.vertices) == pl.nvertices == len(v_expected)
        assert len(pl.segment_lengths) == pl.nsegments == pl.nvertices - 1

        for i, vi in enumerate(v_expected):
            assert pl.vertices[i, 0] == vi[0]
            assert pl.vertices[i, 1] == vi[1]
            if i < pl.nsegments - 1:
                assert pl.segment_lengths[i] == approx(norm(
                    vi[0], vi[1], v_expected[i + 1][0], v_expected[i + 1][1]))
# </Test polyline: Replace vertex>
# #############################################################################


# #############################################################################
# <Test polyline: Distance>
class TestPolylineDistance():
    def test_distance(self):
        v = [[1, 3], [3, 5], [6, 2], [2, 2]]
        pl = Polyline(v)

        # Test values are chosen to yield simple distances and vectors and were
        # calculated by hand apriori
        pts = [[1, 2], [1.5, 5.5], [3, 6], [3.5, 3.5],
               [5.5, 3.5], [4, 1], [2, 2.5]]
        proj_vecs, dists, proj_to = pl.get_dist_to_line(pts)

        # Should show from pt to closest part of the polyline
        proj_vecs_exp = [[0, 1], [1, -1], [0, -1], [0.5, 0.5],
                         [-0.5, -0.5], [0, 1], [0, -0.5]]
        dists_exp = [1., sqrt(2.), 1., sqrt(2.) / 2., sqrt(2.) / 2., 1., 0.5]
        # id < 0: Onto segment with -id-1, else onto vertex with id
        # Last point projects exactly orthogonally to end of segment, thus -3
        proj_to_exp = [0, -1, 1, -2, -2, -3, -3]

        for i, (pvexp, dexp, ptexp) in enumerate(zip(
                proj_vecs_exp, dists_exp, proj_to_exp)):
            assert proj_vecs[i, 0] == approx(pvexp[0])
            assert proj_vecs[i, 1] == approx(pvexp[1])
            assert dists[i] == approx(dexp)
            assert proj_to[i] == ptexp
# </Test polyline: Distance>
# #############################################################################


# #############################################################################
# <Test polyline: Angle>
class TestPolylineAngle():
    def test_angle(self):
        # Test values chosen to yield simple angles: 180°, 135°, 90°, 45°, 0°
        v = [[0, 0], [1, 1], [2, 2], [2, 3], [1, 3], [2, 4], [1, 3]]
        pl = Polyline(v)

        angles = [pl.get_angle_at_vertex(i) / pi * 180.
                  for i in range(1, pl.nsegments)]
        angles_exp = [180., 135., 90., 45., 0.]

        for ang, ang_exp in zip(angles, angles_exp):
            print(ang, ang_exp)
            assert ang == approx(ang_exp, abs=1e-5)
# </Test polyline: Angle>
# #############################################################################