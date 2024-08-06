from qat.purr.utils.algorithm import stable_partition


def test_stable_partition():
    original = [0, 0, 3, -1, 2, 4, 5, 0, 7]
    print(original[-1:])
    partitioned = stable_partition(original, lambda x: x > 0)
    assert partitioned == ([3, 2, 4, 5, 7], [0, 0, -1, 0])
