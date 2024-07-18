"""
Routines for summarising processes and their capability.
"""

from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import scipy
import scipy.stats as st
import seaborn as sns

import mqr

from .summary import Study, Sample

@dataclass
class Specification:
    """
    Process specification.

    Arguments
    ---------
    target (float) -- Design value for the process.
    lsl (float) -- Lower specification limit.
    usl (float) -- Upper specification limit.
    """
    target: float
    lsl: float
    usl: float

@dataclass
class Capability:
    """
    Process capability values.

    Construct this object with a sample and a `mqr.process.Specification`:
    >>> Capability(sample=..., spec=Specification(...))

    Arguments
    ---------
    cp (float) -- Process potential. The capability of the process if it was
        centred at `Specification.target`.
    cpk (float) -- Process capability. The number of standard deviations of
        process variation that fit in the specification, normalised by 3*sigma.
        Ie. a 6-sigma process has capability 2.0.
    defects_st (float) -- Short-term defect rate, based on a fitted normal distribution.
    defects_lt (float) -- Long-term defect rate, based on a normal distribution
        with 1.5*stddev larger than short-term.
    """
    cp: float
    cpk: float
    defects_st: float
    defects_lt: float

    def __init__(self, sample: Sample, spec: Specification):
        """
        Construct Capability.

        Arguments
        ---------
        sample (mqr.process.Sample) -- Set of measurements from KPI.
        spec (mqr.process.Specification) -- Specificatino for KPI.
        """
        self.cp = (spec.usl - spec.lsl) / (6 * sample.std)
        self.cpk = np.minimum(spec.usl - sample.mean, sample.mean - spec.lsl) / (3 * sample.std)
        in_spec = np.logical_and(sample.data >= spec.lsl, sample.data <= spec.usl)
        dist = st.norm(sample.mean, sample.std)
        dist_lt = st.norm(sample.mean, 1.5 * sample.std)
        self.defects_st = 1 - (dist.cdf(spec.usl) - dist.cdf(spec.lsl))
        self.defects_lt = 1 - (dist_lt.cdf(spec.usl) - dist_lt.cdf(spec.lsl))

@dataclass
class Process:
    """
    Model of a process based on study data and specifications.

    Contains a set of statistics including capability for multiple product KPIs
    or multiple stages in a process.

    Construct this object using a `mqr.summary.Study`, a dict that maps
    KPI names to their `mqr.process.Specification`:
    >>> study = mqr.summary.Study(...)
    >>> kpi1 = mqr.process.Specification(...)
    >>> kpi2 = mqr.process.Specification(...)
    >>> specs = {'KPI1': kpi1, 'KPI2': kpi2}
    >>> Process(study, specs)
    """
    study: mqr.summary.Study
    specifications: dict[str, Specification]
    capabilities: dict[str, Capability]

    def __init__(self, study: Study, specifications: dict[str, Specification]):
        """
        Construct Process.

        Arguments
        ---------
        study (mqr.summary.Study) -- Data from process samples.
        specifications (dict[str, Specification]) -- Dict of specifications with
            a spec for every KPI column in the `study`.
        """

        if not set(study.samples.keys()) <= set(specifications.keys()):
            raise ValueError('All samples in study must have a specification.')

        self.study = study
        self.specifications = specifications

        self.capabilities = {
            name: Capability(sample, specifications[name])
            for name, sample
            in study.samples.items()}

    def _repr_html_(self):
        return html(self)

def html(p: Process, prec=3):
    def join(s):
        return ''.join(s)

    def th(scope='col'):
        def _th(s):
            return f'<th scope="{scope}">{s}</th>'
        return _th

    def td(s):
        return f'<td>{s}</td>'

    def bold(s):
        return f'<b>{s}</b>'

    def gray(s):
        return f'<font color="gray">{s}</font>'

    def fmt(value, prec=3):
        return f'{value:#.{prec}g}'

    def compose(f, g):
        return lambda *a, **kw: f(g(*a, **kw))

    names = [name for name in p.study.samples]
    specs = [p.specifications[name] for name in names]
    capabilities = [p.capabilities[name] for name in names]

    return f'''
    <table>
        <caption>Process - {p.study.name}</caption>
        <thead>
            <tr>
                <th scope="col"></th>
                {join([th()(n) for n in names])}
            </tr>
        </thead>
        <tbody>
            <tr>
                <th scope="row"><font color="gray">USL</font></th>
                {join([td(gray(fmt(s.usl, prec))) for s in specs])}
            </tr
            <tr>
                <th scope="row">Target</th>
                {join([td(fmt(s.target, prec)) for s in specs])}
            </tr>
            <tr>
                <th scope="row"><font color="gray">LSL</font></th>
                {join([td(gray(fmt(s.lsl, prec))) for s in specs])}
            </tr>

            <thead><tr></tr></thead>
            <tr>
                <th scope="row"><b>C<sub>pk</sub></b></th>
                {join([td(bold(fmt(s.cpk))) for s in capabilities])}
            </tr>
            <tr>
                <th scope="row">C<sub>p</sub></th>
                {join([td(fmt(s.cp)) for s in capabilities])}
            </tr>
            <tr>
                <th scope="row">Defects<sub>st</sub> (ppm)</th>
                {join([td(fmt(s.defects_st*1e6)) for s in capabilities])}
            </tr>
            <tr>
                <th scope="row">Defects<sub>lt</sub> (ppm)</th>
                {join([td(fmt(s.defects_lt*1e6)) for s in capabilities])}
            </tr>
        <tbody>
        <tfoot>
        </tfoot>
    </table>
    '''
