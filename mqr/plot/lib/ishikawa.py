"""
Fishbone/Ishikawa diagrams for modelling root-causes.
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

def ishikawa(problem:str, causes:dict[str,list[str]], ax, conf:dict=None):
    """
    Draws an Ishikawa/fishbone diagram.

    Draws the "spine", and other levels of "bones". Express the label for the spine
    as the `problem` paramerter. Express the two levels of causes as a dictionary
    of lists: the key to each dictionary names the first branch from the spine,
    the list of values name secondary branches.

    Arguments
    ---------
    problem (str) -- Label for the main spine.
    causes (dict[str, list[str]]) -- Mapping from string values which are labels for the
        first branches to list-of-string keys which are labels for the next
        level causes.
    ax (matplotlib.axes.Axes) -- Axes for the plot.

    Optional
    --------
    conf (dict) -- Keys overriding the default config.

    Examples
    --------
    This example shows a diagram with the main spine labelled as "Problem". Two
    branches leave the spine, then other branches leave those.
    >>> problem = 'Problem'
    >>> causes = {
        'Group1': ['Cause1a', 'Cause1b', 'Cause1c'],
        'Group2': ['Cause2a', 'Cause2b'],}
    """
    conf_update = conf
    conf = default_ishikawa_config()
    if conf_update is not None:
        conf.update(conf_update)

    pad_t, pad_r, pad_b, pad_l = conf['padding']

    bones = list(causes.keys())
    max_causes = max([len(v) for v in causes.values()])
    N = int(np.ceil(len(bones) / 2))

    bone_rise = conf['bone_rise'] = (max_causes + 1) * conf['cause_space']
    head_space = conf['head_space']
    bone_run = conf['bone_run'] = conf['bone_rise'] / np.tan(conf['bone_angle'])
    bone_space = conf['bone_space']
    body_length = (N - 1) * bone_space

    _draw_spine(body_length, head_space, conf=conf, ax=ax)
    ax.text(head_space + 0.2, 0, problem,
            **conf['defect_font_dict'], va='center', ha='left')

    for i in range(N):
        _draw_bone(
            bones[2*i],
            x_start=-i*bone_space,
            bone_rise=bone_rise,
            bone_run=bone_run,
            bone_drn=1,
            conf=conf,
            ax=ax)
        top_causes = causes[bones[2*i]]
        cause_start = (max_causes - len(top_causes)) / 2
        for j, cause in enumerate(top_causes):
            _draw_primary_cause(cause, i, 1, cause_start+j, conf=conf, ax=ax)

        bone_text = '' if (len(bones) <= 2*i+1) else bones[2*i+1]
        _draw_bone(
            bone_text,
            x_start=-i*bone_space,
            bone_rise=bone_rise,
            bone_run=bone_run,
            bone_drn=-1,
            conf=conf,
            ax=ax)

        try:
            bottom_causes = causes[bones[2*i+1]]
            cause_start = (max_causes - len(bottom_causes)) / 2
            for j, cause in enumerate(bottom_causes):
                _draw_primary_cause(cause, i, -1, cause_start+j, conf=conf, ax=ax)
        except:
            pass

    ax.set_aspect('equal', 'box')
    ax.axis('off')

def _draw_spine(body_length, head_space, *, conf, ax):
    x_left, x_right = -body_length, head_space
    y = 0.0
    ax.plot([x_left, x_right], [y, y], **conf['line_kwargs'])

def _draw_bone(text, x_start, bone_rise, bone_run, bone_drn, *, conf, ax):
    x_end = x_start - bone_run
    y_start, y_end = 0.0, bone_drn * bone_rise
    ax.plot([x_start, x_end], [y_start, y_end], **conf['line_kwargs'])
    text_offset = 0.2 * bone_drn
    va = 'bottom' if bone_drn == 1 else 'top'
    ax.text(x_end, y_end+text_offset, text, conf['cause_font_dict'], va=va, ha='center')

def _draw_primary_cause(text, bone_index, bone_drn, height, *, conf, ax):
    bone_start = -bone_index * conf['bone_space']
    y = (height + 1) * conf['cause_space']
    x_right = bone_start - y * np.tan(conf['bone_angle'])
    x_left = bone_start - y * np.tan(conf['bone_angle']) - conf['cause_length']

    ax.annotate(text,
                xy=(x_right, bone_drn * y),
                ha='right',
                va='center',
                xytext=(x_left, bone_drn * y),
                **conf['primary_font_dict'],
                xycoords='data',
                textcoords='data',
                arrowprops=dict(arrowstyle="->",
                                facecolor='black'))

def default_ishikawa_config():
    return {
        'head_space': 2.0,
        'bone_space': 8.0, # NB: nikxn: should be auto based on widest text
        'bone_angle': np.pi / 4,
        'cause_space': 1.0,
        'cause_length': 2.0,
        'padding': (1.0, 1.0, 1.0, 1.0),
        'line_kwargs': {'linewidth': 1.0, 'color': 'k'},
        'defect_font_dict': {'fontsize': 12, 'weight': 'bold'},
        'cause_font_dict': {'fontsize': 9, 'weight': 'bold'},
        'primary_font_dict': {'fontsize': 9},
    }
