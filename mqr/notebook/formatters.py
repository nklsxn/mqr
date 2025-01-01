import mqr

from collections.abc import Iterable
import IPython.core.formatters
import pandas as pd

class HTMLFormatter(IPython.core.formatters.HTMLFormatter):
    def __call__(self, obj):
        if isinstance(obj, mqr.process.Sample):
            return format_samples((obj,))

        elif isinstance(obj, mqr.process.Summary):
            return format_samples(obj.samples.values())

        elif is_iterable_of(obj, mqr.process.Sample):
            return format_samples(obj)

        elif is_dict_of(obj, mqr.process.Sample):
            return format_samples(obj.values())

        elif isinstance(obj, mqr.process.Capability):
            return format_capabilities((obj,))

        elif is_iterable_of(obj, mqr.process.Capability):
            return format_capabilities(obj)

        elif is_dict_of(obj, mqr.process.Capability):
            return format_capabilities(obj.values())

        else:
            return super().__call__(obj)

def is_iterable_of(obj, typ):
    return (
        isinstance(obj, Iterable) and
        all([isinstance(elem, typ) for elem in obj])
    )

def is_dict_of(obj, typ):
    return (
        isinstance(obj, dict) and
        all([isinstance(elem, typ) for elem in obj.values()])
    )

def format_samples(samples):
    display_fmt = mqr.notebook.Defaults.sample_value_fmt

    def join(s):
        return ''.join(s)

    def th(scope='col'):
        def _th(s):
            return f'<th scope="{scope}">{s}</th>'
        return _th

    def td(s):
        return f'<td>{s}</td>'

    def fmt_value(value):
        return f'{value:{display_fmt}}'
    
    col_headers = [s.name for s in samples]

    ad_stat = [s.ad_stat for s in samples]
    ad_pvalue = [s.ad_pvalue for s in samples]
    ks_stat = [s.ks_stat for s in samples]
    ks_pvalue = [s.ks_pvalue for s in samples]

    nobs = [s.nobs for s in samples]
    mean = [s.mean for s in samples]
    std = [s.std for s in samples]
    var = [s.var for s in samples]
    skewness = [s.skewness for s in samples]
    kurtosis = [s.kurtosis for s in samples]
    minimum = [s.minimum for s in samples]
    quartile1 = [s.quartile1 for s in samples]
    median = [s.median for s in samples]
    quartile3 = [s.quartile3 for s in samples]
    maximum = [s.maximum for s in samples]

    outliers = [len(s.outliers) for s in samples]

    html = f'''
    <table>
        <thead>
            <tr>
                <th scope="col"></th>
                {join(map(th(), col_headers))}
            </tr>
        </thead>
        <tbody>
            <tr>
                <th colspan={len(col_headers)+1} style="text-align:left;">Normality (Anderson-Darling)</th>
            </tr>
            <tr>
                <th scope="row">Stat</th>
                {join(map(td, map(fmt_value, ad_stat)))}
            </tr>
            <tr>
                <th scope="row">P-value</th>
                {join(map(td, map(fmt_value, ad_pvalue)))}
            </tr>

            <thead><tr></tr></thead>
            <tr>
                <th scope="row">N</th>
                {join(map(td, nobs))}
            </tr>

            <thead><tr></tr></thead>
            <tr>
                <th scope="row">Mean</th>
                {join(map(td, map(fmt_value, mean)))}
            </tr>
            <tr>
                <th scope="row">StdDev</th>
                {join(map(td, map(fmt_value, std)))}
            </tr>
            <tr>
                <th scope="row">Variance</th>
                {join(map(td, map(fmt_value, var)))}
            </tr>
            <tr>
                <th scope="row">Skewness</th>
                {join(map(td, map(fmt_value, skewness)))}
            </tr>
            <tr>
                <th scope="row">Kurtosis</th>
                {join(map(td, map(fmt_value, kurtosis)))}
            </tr>

            <thead><tr></tr></thead>
            <tr>
                <th scope="row">Minimum</th>
                {join(map(td, map(fmt_value, minimum)))}
            </tr>
            <tr>
                <th scope="row">1st Quartile</th>
                {join(map(td, map(fmt_value, quartile1)))}
            </tr>
            <tr>
                <th scope="row">Median</th>
                {join(map(td, map(fmt_value, median)))}
            </tr>
            <tr>
                <th scope="row">3rd Quartile</th>
                {join(map(td, map(fmt_value, quartile3)))}
            </tr>
            <tr>
                <th scope="row">Maximum</th>
                {join(map(td, map(fmt_value, maximum)))}
            </tr>

            <thead><tr></tr></thead>
            <tr>
                <th scope="row">N Outliers</th>
                {join(map(td, outliers))}
            </tr>
        </tbody>
        <tfoot>
        </tfoot>
    </table>
    '''
    return html

def format_capabilities(capabilities):
    display_fmt = mqr.notebook.Defaults.capability_value_fmt

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

    def fmt(value):
        return f'{value:{display_fmt}}'

    def compose(f, g):
        return lambda *a, **kw: f(g(*a, **kw))

    names = [c.sample.name for c in capabilities]
    specs = [c.spec for c in capabilities]

    return f'''
    <table>
        <thead>
            <tr>
                <th scope="col"></th>
                {join([th()(n) for n in names])}
            </tr>
        </thead>
        <tbody>
            <tr>
                <th scope="row"><font color="gray">USL</font></th>
                {join([td(gray(fmt(s.usl))) for s in specs])}
            </tr
            <tr>
                <th scope="row">Target</th>
                {join([td(fmt(s.target)) for s in specs])}
            </tr>
            <tr>
                <th scope="row"><font color="gray">LSL</font></th>
                {join([td(gray(fmt(s.lsl))) for s in specs])}
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
