from __future__ import annotations
from icon4py.model.common.test_utils.datatest_fixtures import grid_savepoint
import numpy as np
# from icon4py.model.common.grid import IconGrid

from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common import dimension as dims


def connectivity(label: str, icon_grid):
    offset_provider = icon_grid.offset_provider_mapping[label]
    return offset_provider[0](*offset_provider[1:]).table


def has_skip_values(conn):
    return np.any(conn == -1)


class HashableArray:
    def __init__(self, arr: np.ndarray):
        self.arr = arr
        self.hash = hash(arr.data.tobytes())

    def __hash__(self):
        return self.hash

    def __eq__(self, other: HashableArray):
        return np.array_equal(self.arr, other.arr)


class StranglyHashableArray:
    def __init__(self, arr: np.ndarray):
        self.arr = arr

    def __hash__(self):
        return 0

    def __eq__(self, other: StranglyHashableArray):
        # ignore -1 (skip) values
        mask_self = np.equal(self.arr, -1)
        mask_other = np.equal(other.arr, -1)
        mask = np.logical_or(mask_self, mask_other)
        clean_self = np.where(mask, 0, self.arr)
        clean_other = np.where(mask, 0, other.arr)
        return np.array_equal(clean_self, clean_other)


def compress(first, *other, elem=0):
    composed = first
    for o in other:
        composed = o[composed]
    _, ind, inv = np.unique(composed[elem], return_index=True, return_inverse=True)
    compressor = np.unravel_index(ind, composed[elem].shape)
    decompressor = inv.reshape(composed[elem].shape)
    # print(len(ind))
    # print(inv)
    label = {}
    labels = iter(range(0, len(inv)))
    res = []
    for i in inv:
        if i in label:
            res.append(label[i])
        else:
            l = next(labels)
            res.append(l)
            label[i] = l
    # print("".join(str(r) for r in res))

    # 1 0 3 -> 0 1 2
    # 2 3 0 -> 0 2 1
    # 3 2 1 -> 1 2 0
    # if res[3:] == [1, 0, 3]:
    #     reorder = np.array([0, 1, 2])
    # elif res[3:] == [2, 3, 0]:
    #     reorder = np.array([0, 2, 1])
    # elif res[3:] == [3, 2, 1]:
    #     reorder = np.array([1, 2, 0])
    # else:
    #     reorder = np.array([0, 1, 2])
    #     # print(np.array(res).reshape(composed[elem].shape))
    #     # raise AssertionError("unknown pattern")
    # other[0][elem, :] = other[0][elem, :][reorder]

    print(np.array(res).reshape(composed[elem].shape))
    # print(f"Compressed to {len(ind)} elements with {compressor}.")
    return composed[(slice(None), *compressor)], decompressor


def analyze(first, other):  # TODO extend to more
    stack = []
    for i in range(first.shape[1]):
        for j in range(other.shape[1]):
            stack.append(other[:, j][first[:, i]])
            print(f"{i}/{j}: {other[:, j][first[:, i]]}")
    s = set(HashableArray(s) for s in stack)
    print(f"Compressed from {len(stack)} to {len(s)} elements.")


def analyze_skip_value(first, other):  # TODO extend to more
    stack = []
    for i in range(first.shape[1]):
        for j in range(other.shape[1]):
            tmp = other[:, j][first[:, i]]
            tmp[first[:, i] == -1] = -1  # fixes places where we did wrap-around indexing with -1
            print(f"{i}/{j}: {tmp}")
            stack.append(tmp)
    s = set(StranglyHashableArray(s) for s in stack)
    print(f"Compressed from {len(stack)} to {len(s)} elements.")


from itertools import count


def find_permutation(a, b):
    # assume only col permutations are good enough
    perm = []
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            if np.array_equal(a[:, i], b[:, j]):
                perm.append(j)
    return perm


def relabel(tmp):
    unique, counts = np.unique(tmp, return_counts=True)
    unique, counts = unique[np.argsort(-counts)], counts[np.argsort(-counts)]  # order by frequency
    one_lbl = 2
    two_lbl = 0
    labels = {}
    # TODO generalize
    for u, c in zip(unique, counts):
        if c == 1:
            labels[u] = one_lbl
        else:
            labels[u] = two_lbl
            two_lbl += 1

    for i in range(tmp.shape[0]):
        for j in range(tmp.shape[1]):
            tmp[i, j] = labels[tmp[i, j]]
    return tmp


def analyze2(first, second):  # e2c c2v -> e x e2c x c2v
    composed = second[first]
    new_c2v = np.empty_like(second)
    for i in range(composed.shape[0]):
        print(f"{relabel(composed[i])}")
        # perm = find_permutation(composed[0], composed[i])
        # new_c2v[?]


def invert(c):
    c, ind, inv = np.unique(c, return_index=True, return_inverse=True)


def analyze3(e2c, c2v):
    for e in range(e2c.shape[0]):
        cs = e2c[e]

        print(f"{relabel(c2v[cs])}")


def test_connectivity_composition(icon_grid: "IconGrid"):
    c2e = connectivity("C2E", icon_grid)
    assert not has_skip_values(c2e)
    e2v = connectivity("E2V", icon_grid)
    assert not has_skip_values(e2v)
    c2v = connectivity("C2V", icon_grid)
    assert not has_skip_values(c2v)
    v2e = connectivity("V2E", icon_grid)
    assert not has_skip_values(v2e)
    e2c = connectivity("E2C", icon_grid)

    # e2c2v = connectivity("E2C2V", icon_grid)

    # c2e2cO = connectivity("C2E2CO", icon_grid)
    # cell_domain = h_grid.domain(dims.CellDim)(h_grid.Zone.INTERIOR)
    # c_start, c_end = icon_grid.start_index(cell_domain), icon_grid.end_index(cell_domain)
    # c2e2cO_interior = c2e2cO[c_start:c_end, :]
    # assert not has_skip_values(c2e2cO_interior)
    # print("foo")
    # analyze(c2e2cO_interior, c2e2cO)
    # exit(1)

    # analyze(c2e, e2v)
    # analyze(c2v, v2e)
    # compress(c2v, v2e, e2v)
    # analyze_skip_value(e2c, c2v)

    edge_domain = h_grid.domain(dims.EdgeDim)(h_grid.Zone.INTERIOR)
    start = icon_grid.start_index(edge_domain)
    end = icon_grid.end_index(edge_domain)
    e2c_interior = e2c[start:end, :]
    assert not has_skip_values(e2c_interior)
    # analyze(e2c_interior, c2v)
    analyze3(e2c_interior, c2v)

    # assert np.array_equal(e2v[start:end, :], e2c2v[start:end, :2])
    # print(e2c_interior.shape[0])
    # for i in range(e2c_interior.shape[0]):
    #     compress(e2c_interior, c2v, elem=i)
