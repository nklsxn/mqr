import mqr.inference.lib.util as util

def alternative(alt, lib):
    """
    Convert an alternative string from the scipy convention to another convention.

    Parameters
    ----------
    alt : {'two-sided', 'greater', 'less'}
        Sense of alternative hypothesis to convert.
    lib : {'statsmodels'}
        Library whose convention to convert to. Only 'statsmodels' currently used.

    Returns
    -------
    str
        Sense of alternative expressed in convention of ``lib``.
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
    Convert bounds argument of a confidence interval to another library's convention.

    Parameters
    ----------
    bounded : {'both', 'above', 'below'}
        Which side of the confidence interval is bounded.
    lib : {'statsmodels', 'scipy'}
        Convert to this library's convention.

    Returns
    -------
    str
        Sense of 'alternative' expressed in convention of `lib`.
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
