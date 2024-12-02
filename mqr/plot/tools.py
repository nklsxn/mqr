"""
======================================
Plotting tools (:mod:`mqr.plot.tools`)
======================================

.. currentmodule:: mqr.plot.tools

.. rubric:: Functions
.. autosummary::
    :toctree: generated/

    sharex
    sharey
"""

def sharey(fig, axs, remove_space=True):
    for ax in axs:
        if ax != axs[0]:
            ax.sharey(axs[0])
        ax.label_outer(remove_inner_ticks=remove_space)
        if remove_space:
            fig.get_layout_engine().set(w_pad=0, wspace=0)

def sharex(fig, axs, remove_space=True):
    for ax in axs:
        if ax != axs[0]:
            ax.sharex(axs[0])
        ax.label_outer(remove_inner_ticks=remove_space)
        if remove_space:
            fig.get_layout_engine().set(h_pad=0, hspace=0)
