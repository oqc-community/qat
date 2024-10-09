def stable_partition(iterable, unary_pred):
    """
    A brute force implementation of the C++ STL stable_partition algorithm.
    TODO - research and improve implementation to become O(n*log(n))
    TODO - generalise to multiple predicate filtering while preserving relative order
    """

    head = []
    tail = []
    for elt in iterable:
        if unary_pred(elt):
            head.append(elt)
        else:
            tail.append(elt)
    return head, tail
