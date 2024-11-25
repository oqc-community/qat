def stable_partition(iterable, unary_pred):
    """
    A brute force implementation of the C++ STL stable_partition algorithm. Instead of in-place
    rearrangement of elements, it creates new structures.
    TODO - If we wanted to fully honour the original implementation, how could we make it O(n*log(n)) ?
    TODO - Might be interesting to generalise to multiple predicate filters while preserving relative order
    """

    head = []
    tail = []
    for elt in iterable:
        if unary_pred(elt):
            head.append(elt)
        else:
            tail.append(elt)
    return head, tail
