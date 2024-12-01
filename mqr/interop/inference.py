import mqr.inference.lib.util as util

def alternative(alt, lib):
    """
    Convert an alternative string from the scipy convention to another convention.

    Arguments
    ---------
    alt (str) -- Sense of alternative hypothesis to convert. Must be one of
        "two-sided", "greater" or "less".
    lib (str) -- Library whose convention to convert to.

    Returns
    -------
    (str) -- Sense of alternative expressed in convention of `lib`.
    """
    if lib == 'statsmodels':
        if alt == 'two-sided':
            return 'two-sided'
        elif alt == 'less':
            return 'smaller'
        elif alt == 'greater':
            return 'larger'
        else:
            raise ValueError(util.alternative_error_msg(alt))
    raise ValueError(f'Invalid library {lib}.')

def bounded(bounded, lib):
    """
    Convert argument specifying bounds of a confidence interval to another
    library's convention.

    Libraries use the term "alternative" (from alternative hypothesis), when
    this feature is available. The bounds in a confidence interval are related
    to the sense of the alternative in an hypothesis test. When the alternative
    is "less", the corresponding confidence interval includes the right tail and
    is bounded "below", and vice-versa.

    Arguments
    ---------
    bounded (str) -- Which side of the confidence interval is bounded. Must be
        one of "both", "above" or "below".
    lib (str) -- Convert to this library's convention.

    Returns
    -------
    (str) -- Sense of "alternative" expressed in convention of `lib`.
    """
    if lib == 'statsmodels':
        if bounded == 'both':
            return 'two-sided'
        elif bounded == 'below':
            return 'larger'
        elif bounded == 'above':
            return 'smaller'
        else:
            raise ValueError(util.bounded_error_msg(bounded))
    elif lib == 'scipy':
        if bounded == 'both':
            return 'two-sided'
        elif bounded == 'below':
            return 'greater'
        elif bounded == 'above':
            return 'less'
        else:
            raise ValueError(util.bounded_error_msg(bounded))
    else:
        raise ValueError(f'Invalid library {lib}.')
