"""
===================================
Notebook tools (:mod:`mqr.nbtools`)
===================================

Tools for presenting results in notebooks.

.. rubric:: Functions

.. autosummary::
    :toctree: generated/

    grab_figure
    hstack
    vstack
"""

from base64 import b64encode
from enum import Enum
import IPython
from IPython.core.pylabtools import print_figure
from IPython.display import display, HTML
import markdown
from matplotlib._pylab_helpers import Gcf

Line = Enum('Line', ['HORIZONTAL', 'VERTICAL'])

def set_container_width(width_pct: int):
    """
    Changes the Jupyter notebook HTML container width.

    Useful on widescreens to use available screen space.

    Parameters
    ----------
    width_pct : int
        Expand the notebook width to this much of the view. Eg. to take up 95%
        of the page, use `width_pct=95`.

    """
    assert width_pct > 0 and width_pct <= 100, f'Argument `width_pct={width_pct}` must be 0<width_pct<=100'
    display(HTML(f'<style>.container {{ width:{width_pct}% !important; }}</style>'))

def grab_figure(figure, suppress=True):
    '''
    Renders the figure to png, base-64 encoded HTML img tag.

    The resulting HTML will be rendered if it is the last expression in a cell.
    It can also be passed to :func:`hstack` or :func:`vstack` to arrange plots
    around other outputs.

    Parameters
    ----------
    figure : matplotlib.figure.Figure
        The figure to capture.
    suppress : bool, optional
        Suppresses the figure by destroying the object. Since this function is
        used to control how the figure is displayed, the default is to destroy
        the object after capturing its output.
    '''
    raw_data_b64 = b64encode(print_figure(figure)).decode('utf-8')
    image_data = f'data:image/png;base64,{raw_data_b64}'
    if suppress:
        Gcf.destroy_fig(figure)
    return HTML(f'<img src="{image_data:s}" />')

def hstack(*args, margin='5px 10px 5px 10px',
        justify_content='stretch', justify_items='center',
        align_content='start', align_items='start'):
    '''
    Horizontally stack the html, markdown or string representation of `args`.

    The `args` will be stacked in order from left to right, and converted like this:

    * if the arg has the attribute `__repr__`, then its output is used pre-formatted,
    * if the arg has the attribute `_repr_html_`, then its output is used,
    * if the arg is a string, then it is treated as markdown and converted to HTML,
    * otherwise, the arg's default string conversion is used.

    Parameters
    ----------
    args
        Elements to stack horizontally.
    margin : float
        Set in the style attribute of a div wrapping elements in this stack
        (excluding lines, which have a fixed 5px margin).
    justify_content : str, optional
        Set in the style attribute of the flexbox containing this stack.
    justify_items : str, optional
        Set in the style attribute of the flexbox containing this stack.
    align_content : str, optional
        Set in the style attribute of the flexbox containing this stack.
    align_items : str, optional
        Set in the style attribute of the flexbox containing this stack.

    Returns
    -------
    IPython.display.HTML
        Stacked elements which can be directly displayed in jupyter.
    '''
    jc = f'justify-content:{justify_content};'
    ji = f'justify-items:{justify_items};'
    ac = f'align-content:{align_content};'
    ai = f'align-items:{align_items};'

    raw_html = f'<div style="display:flex;flex-direction:row;{jc}{ji}{ac}{ai}">'
    for elem in args:
        raw_html += _to_html(elem, margin)
    raw_html += '</div>'
    return HTML(raw_html)

def vstack(*args, margin='5px 10px 5px 10px',
        justify_content='start', justify_items='start',
        align_content='start', align_items='start'):
    '''
    Vertically stack the html, markdown or string representation of `args`.

    The `args` will be stacked in order from top to bottom, and converted like this:

    * if the arg has the attribute `__repr__`, then its output is used pre-formatted,
    * if the arg has the attribute `_repr_html_`, then its output is used,
    * if the arg is a string, then it is treated as markdown and converted to HTML,
    * otherwise, the arg's default string conversion is used.

    Parameters
    ----------
    args
        Elements to stack vertically.

    margin : str, optional
        Set in the style attribute of a div wrapping elements in this stack
        (excluding lines, which have a fixed 5px margin). Give values in the
        order, eg. "5px 10px 5px 10px" (top right bottom left).
    justify_content, justify_items, align_content, align_items : str
        Set in the style attribute of the flexbox containing this stack.

    Returns
    -------
    IPython.display.HTML
        Stacked elements which can be directly displayed in jupyter.
    '''
    jc = f'justify-content:{justify_content};'
    ji = f'justify-items:{justify_items};'
    ac = f'align-content:{align_content};'
    ai = f'align-items:{align_items};'

    raw_html = f'<div style="display:flex;flex-direction:column;{jc}{ji}{ac}{ai}">'
    for elem in args:
        raw_html += _to_html(elem, margin)
    raw_html += '</div>'
    return HTML(raw_html)

def _to_html(arg, margin):
    fmt = IPython.get_ipython().display_formatter.formatters['text/html']
    def div(elem):
        return f'<div style="margin:{margin};">' + elem + '</div>'

    if isinstance(arg, Line):
        if arg is Line.HORIZONTAL:
            return f'<div style="border-top:solid 1px #AAAAAA;margin:5px;align-self:stretch;"></div>'
        elif arg is Line.VERTICAL:
            return f'<div style="border-left:solid 1px #AAAAAA;margin:5px;align-self:stretch;"></div>'

    if type(arg) is str:
        return div(markdown.markdown(arg, extensions=['tables']))

    if fmt is not None:
        result = fmt(arg)
        if result is None:
            result = '<pre>' + repr(arg) + '</pre>'
        return div(result)

    if (fmt is None) or (result is None):
        if hasattr(arg, '_repr_html_'):
            return div(arg._repr_html_())
        elif hasattr(arg, '__repr__'):
            raw = '<pre>' + repr(arg) + '</pre>'
            return div(raw)
        else:
            return div(str(arg))
    else:
        return result
