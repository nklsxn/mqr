Getting Started
===============


## What MQR does

MQR is a toolkit for applied statistics in python.
It implements some statistical tools and brings those together with existing tools from other libraries.
The library provides an organised interface for common activities,
and is designed to ease planning, analysis and reporting in jupyter notebooks
(or IDEs that run code to produce interactive output).

Most of the tools are related to quite general statistical methods that are
commonly used in fields outside design and manufacturing.

The tools in MQR cover four topics:

* process description,
* data analysis and inference,
* design of experiments, and
* statistical quality control.

See the <project:user-guide.md> for notes by topic, and take a look at the
<project:api-ref.rst> for information about each module, class and function.

Originally, MQR stood for _manufacturing quality and reliability_.
MQR doesn't include tools for modelling product lifetime
(see [reliability](inv:reliability:std#index) for that).
However, it does provide the tools to create and analyse experiments that look for improvements in reliability.


## Installation

Install the package using pip:
```bash
pip install mqr-quality
```

Import the package using:
```python
import mqr
```

## What's in MQR

MQR is designed to help engineers improve quality and reliability in products and
the processes that create products. To achieve that, the library does three things:

1.  **Code that automates commonly used plots and tables.**

    Some plots turn up a lot in quality activities. MQR aims to reduce the burden
    of understanding the details of libraries like numpy, pandas, matplotlib, etc,
    and reduce repetitive code when plotting. The goal of these features is to make
    common plotting and descriptive activities fast.

1.  **Provides a uniform interface to functionality from other libraries and MQR.**

    Existing libraries provide good functionality for hypothesis tests, particularly
    numpy, scipy and statsmodels. However, because the tests come from statistical
    literature, and are organised varously by name, purpose or application, they
    can be difficult to navigate, especially with limited experience using python.
    The purpose of wrapping these tools is (1) to organise tests by goal or application
    (eg. tests on means, tests on proportions), and (2) to provide a uniform
    interface that is easy for engineers to navigate and use in jupyter notebooks.

    The <project:#mqr.doe> module is another example. It provides an interface for designing experiments.
    It can be used with [pyDOE3](inv:pyDOE3:std#index) (and provides convenience functions for that),
    but also provides extra features that help with the practicalities of designing
    experiments, collecting and analysing data. For example, experimental designs
    can be built up in a few lines of code by composing smaller sets of runs.

1.  **Code that implements useful features**

    MQR has tools to easily setup crossed GRR studies and the related mixed effects
    analysis. The GRR studies are constructed using tools from [statsmodels](inv:statsmodels:std#index).

    The library also implements hypothesis tests and confidence intervals that are
    not covered in other libraries like some confidence intervals on rates and proportions.

    Statistical quality control tools are also provided, including traditional
    control charts. The interface is easy to extend, so users can easily monitor
    processes with chart types that we haven't implemented yet.


## A note on wrapped interfaces

There are benefits to wrapping other libraries in uniform interfaces when using
a notebook. However, if you use these libraries outside of a notebook, for example
in scripts or automated routines, or if you need more advanced functionality that
is not exposed in the MQR interface, you should look up the original functions
and then call the statistical libraries directly.

Any interface that wraps another library gives details in its docstring,
which is the same text that appears in the <project:api-ref.rst>. Statistical
routines that we implemented are based on various published work; see the
references in docstrings.
