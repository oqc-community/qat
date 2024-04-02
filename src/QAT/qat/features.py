from enum import Flag, auto


class Features(Flag):
    """
    Flags for enabling niche, optional or experimental features that
    by default most people may not want enabled.

    These features can fundamentally change the underlying mechanisms of the project
    so only enable if you're comfortable with what they do.
    """

    # Empty flag for when nothing else is enabled. True/false doesn't matter.
    Empty = auto()

    """
    Replaces basic QIR support with our hybrid runtime, Rasqal. See QAT documentation for full 
    details, but this enables the full QIR spec to be passed, and run, in a hybrid manner. 
    """
    Rasqal = auto()


features_enabled = Features.Empty


def enable_feature(features: Features):
    global features_enabled
    features_enabled = features_enabled | features


def is_rasqal_enabled():
    return Features.Rasqal in features_enabled


def disable_feature(features: Features):
    global features_enabled
    if features in features_enabled:
        features_enabled = features_enabled ^ features


def disable_all_features():
    global features_enabled
    features_enabled = Features.Empty
