"""
Fishbone/Ishikawa diagrams for modelling root-causes.
"""

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from mqr.plot.lib.util import set_kws
from mqr.plot.defaults import Defaults

def ishikawa(problem:str, causes:dict[str,list[str]], ax, ishikawa_kws=None):
    if ishikawa_kws is not None:
        conf = Defaults.ishikawa | ishikawa_kws
    else:
        conf = Defaults.ishikawa

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
