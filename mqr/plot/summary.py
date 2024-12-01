import matplotlib.pyplot as plt
import seaborn as sns

import mqr
from mqr.summary import Sample
from mqr.plot.lib.util import set_kws

def summary(sample: Sample, ax, hyp_mean=None,
            hist_kws=None, box_kws=None, conf_kws=None):
    hist_kws = set_kws(
        hist_kws,
        color='C0',
    )
    box_kws = set_kws(
        box_kws,
        orient='h',
        color='C0',
    )
    conf_kws = set_kws(
        conf_kws,
    )

    ax = ax.flatten()
    if ax.shape != (3,):
        raise ValueError(f'Axes shape must be (3,) but is {ax.shape}.')

    sns.histplot(sample.data, ax=ax[0], **hist_kws)
    sns.boxplot(sample.data, ax=ax[1], **box_kws)
    mqr.plot.confint(sample.conf_mean, ax=ax[2], hyp_value=hyp_mean, **conf_kws)

    ax[1].sharex(ax[0])
    ax[2].sharex(ax[0])
    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.setp(ax[1].get_xticklabels(), visible=False)
    ax[1].tick_params(axis='y', left=False)
    ax[2].set_title('')
    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    ax[2].set_xlabel(sample.name)
