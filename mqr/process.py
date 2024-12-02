"""
===========================================
Process and capability (:mod:`mqr.process`)
===========================================

.. currentmodule:: mqr.process

Routines for summarising processes and their capability.

.. rubric:: Classes

.. autosummary::
    :toctree: generated/

    Process
    Capability
    Specification
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

    Attributes
    ----------
    target : float
        Design value for the process.
    lsl : float
        Lower specification limit.
    usl : float
        Upper specification limit.
    """
    target: float
    lsl: float
    usl: float

@dataclass
class Capability:
    """
    Process capability values.

    Attributes
    ----------
    cp : float
        Process potential. The capability of the process if it was centred at
        `Specification.target`.
    cpk : float
        Process capability. The number of standard deviations of process variation
        that fit in the specification, normalised by 3*sigma. Ie. a 6-sigma
        process has capability 2.0.
    defects_st : float
        Short-term defect rate, based on a fitted normal distribution.
    defects_lt : float
        Long-term defect rate, based on a normal distribution with 1.5*stddev
        larger than short-term.

    Examples
    --------
    Construct this object with a sample and a `mqr.process.Specification`:

    .. code-block:: python
        :emphasize-lines: 3

        sample = mqr.summary.Sample(data['KPI1'])
        spec = mqr.process.Specification(150, 147, 153)
        mqr.process.Capability(sample, spec)

    """
    cp: float
    cpk: float
    defects_st: float
    defects_lt: float

    def __init__(self, sample: Sample, spec: Specification):
        """
        Construct Capability.

        Attributes
        ----------
        sample : mqr.process.Sample
            Set of measurements from KPI.
        spec : mqr.process.Specification
            Specificatino for KPI.

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

    Attributes
    ----------
    study : mqr.summary.Study
        Study object constructed from measured data.
    specification : dict[str, Specification]
        Dictionary of specifications corresponding to the measurements in `study`.
    capabilities : dict[str, Capability], automatic
        Potentials and capabilities of the given measurements. This is automatically
        calculated from `study` and `specification`.

    Examples
    --------
    Construct this object using a `mqr.summary.Study`, a dict that maps KPI
    names to their `mqr.process.Specification`.

    .. code-block:: python
        :emphasize-lines: 10

        study = mqr.summary.Study(data, ['KPI1', 'KPI2', 'KPI3', 'KPO1', 'KPO2'])
        specs = {
            'KPI1': mqr.process.Specification(150, 147, 153),
            'KPI2': mqr.process.Specification(20, 19, 21),
            'KPI3': mqr.process.Specification(15, 14, 16),
            'KPO1': mqr.process.Specification(160, 155, 165),
            'KPO2': mqr.process.Specification(4, 3.5, 4.5),
        }

        mqr.process.Process(study, specs)

    The process produced by that code is shown in jupyter notebooks as an HTML table:

    +-----------------+----------+----------+----------+----------+----------+
    |                 | KPI1     | KPI2     | KPI3     | KPO1     | KPO2     |
    +=================+==========+==========+==========+==========+==========+
    | USL             | 153.     | 21.0     | 16.0     | 165.     | 4.50     |
    +-----------------+----------+----------+----------+----------+----------+
    | Target          | 150.     | 20.0     | 15.0     | 160.     | 4.00     |
    +-----------------+----------+----------+----------+----------+----------+
    | LSL             | 147.     | 19.0     | 14.0     | 155.     | 3.50     |
    +-----------------+----------+----------+----------+----------+----------+
    |                                                                        |
    +-----------------+----------+----------+----------+----------+----------+
    | Cpk             | 0.844    | 1.36     | 0.00155  | 0.806    | 0.103    |
    +-----------------+----------+----------+----------+----------+----------+
    | Cp              | 0.852    | 1.36     | 0.441    | 0.813    | 0.107    |
    +-----------------+----------+----------+----------+----------+----------+
    | Defectsst (ppm) | 1.06e+04 | 45.6     | 5.02e+05 | 1.47e+04 | 7.49e+05 |
    +-----------------+----------+----------+----------+----------+----------+
    | Defectslt (ppm) | 8.83e+04 | 6.57e+03 | 5.38e+05 | 1.04e+05 | 8.31e+05 |
    +-----------------+----------+----------+----------+----------+----------+

    """
    study: mqr.summary.Study
    specifications: dict[str, Specification]
    capabilities: dict[str, Capability]

    def __init__(self, study: Study, specifications: dict[str, Specification]):
        """
        Construct Process.

        Parameters
        ----------
        study : mqr.summary.Study
            Data from process samples.
        specifications : dict[str, Specification]
            Dict of specifications with a spec for every KPI column in the `study`.
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
