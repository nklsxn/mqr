====================================================
Efficieny of estimates of process standard deviation
====================================================

Both R-charts and XBar-charts based on range use sample range to estimate process
standard deviation. Estimates of process standard deviation from sample range are
less efficient than estimates from sample standard deviation when sample size is
greater tha two, and those estimates are equal in efficiency when sample size is
exactly two (see [1]_). Therefore, routines using sample standard deviation are
usually preferable.

The routines using sample range were written for teaching/learning, for times
when the simplicity of working with ranges is more important than the loss in
efficiency, and when trying to reproduce older range-based calculations.

Related classes
---------------
:class:`mqr.spc.XBarParams`, :class:`mqr.spc.RParams`, :class:`mqr.spc.SParams`.

References
----------
.. [1]  Mahmoud, M. A., Henderson, G. R., Epprecht, E. K., & Woodall, W. H. (2010).
        Estimating the standard deviation in quality-control applications.
        Journal of Quality Technology, 42(4), 348-357.
