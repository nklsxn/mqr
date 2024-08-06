"""
Tools for notebooks appearance.
"""

from base64 import b64encode
from enum import Enum
from IPython.core.pylabtools import print_figure
from IPython.display import HTML
import markdown
from matplotlib._pylab_helpers import Gcf

Line = Enum('Line', ['HORIZONTAL', 'VERTICAL'])

def set_container_width(width_pct: int):
    """
    Changes the Jupyter notebook HTML container width.

    Useful on widescreens to use available screen space.

    Arguments
    ---------
    width_pct (int) -- Expand the notebook width to this much of the view. Eg.
        to take up 95% of the page, use `width_pct=95`.
    """
    assert width_pct > 0 and width_pct <= 100, f'Argument `width_pct={width_pct}` must be 0<width_pct<=100'
    display(HTML(f'<style>.container {{ width:{width_pct}% !important; }}</style>'))

def grab_figure(figure, suppress=True):
    '''
    Renders the figure to png, base-64 encoded HTML img tag.

    Arguments
    ---------
    figure (matplotlib.figure.Figure) -- The figure to capture.

    Optional
    --------
    suppress (bool) -- Optionally suppresses the figure by destroying the object.
        Since this function is used to control how the figure is displayed, the
        default is to destroy the object after capturing its output.
    '''
    raw_data_b64 = b64encode(print_figure(figure)).decode('utf-8')
    image_data = f'data:image/png;base64,{raw_data_b64}'
    if suppress:
        Gcf.destroy_fig(figure)
    return HTML(f'<img src="{image_data:s}" />')

def hstack(*args, margin=5):
    '''
    Horizontally stack the html, markdown or string representation of `args`.

    The `args` will be stacked in order from left to right, and converted like this:
    * if the arg has the attribute `_repr_html_`, then its output is used,
    * if the arg is a string, then it is treated as markdown and converted to HTML,
    * otherwise, the arg's default string conversion is used.

    Arguments
    ---------
    args -- Elements to stack horizontally.

    Returns
    -------
    (IPython.display.HTML) -- the stacked elements which can be directly
        displayed in jupyter.
    '''
    raw_html = '<div style="display:flex; flex-direction:row; align-items:stretch;">'
    for elem in args:
        raw_html += _to_html(elem, margin)
    raw_html += '</div>'
    return HTML(raw_html)

def vstack(*args, margin=5):
    '''
    Vertically stack the html, markdown or string representation of `args`.

    The `args` will be stacked in order from top to bottom, and converted like this:
    * if the arg has the attribute `_repr_html_`, then its output is used,
    * if the arg is a string, then it is treated as markdown and converted to HTML,
    * otherwise, the arg's default string conversion is used.

    Arguments
    ---------
    args -- Elements to stack vertically.

    Returns
    -------
    (IPython.display.HTML) -- the stacked elements which can be directly
        displayed in jupyter.
    '''
    raw_html = '<div style="display:flex; flex-direction:column; align-items:stretch;">'
    for elem in args:
        raw_html += _to_html(elem, margin)
    raw_html += '</div>'
    return HTML(raw_html)

def _to_html(arg, margin):
    def div(elem):
        return f'<div style="margin:{margin}px;">' + elem + '</div>'

    if hasattr(arg, '_repr_html_'):
        return div(arg._repr_html_())
    elif isinstance(arg, Line):
        if arg is Line.HORIZONTAL:
            return f'<div style="border-top:solid 1px #AAAAAA; margin:{margin/2}px;"></div>'
        elif arg is Line.VERTICAL:
            return f'<div style="border-left:solid 1px #AAAAAA; margin:{margin/2}px;"></div>'
    elif type(arg) is str:
        return div(markdown.markdown(arg, extensions=['tables']))
    else:
        return div(str(arg))
