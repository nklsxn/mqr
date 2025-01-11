Efficieny of estimates of process standard deviation
====================================================

Both R-charts and XBar-charts can use sample range to estimate process standard
deviation. Estimates of process standard deviation from sample range are less
efficient than estimates from sample standard deviation when sample size is
greater than two, and those estimates are equal in efficiency when sample size is
exactly two (see [^1]). Therefore, routines using sample standard deviation might
be preferable.

The routines using sample range were written for teaching/learning, for times
when interpretation of range is easier than standard deviation, when the
simplicity of working with ranges is more important than the loss in efficiency
(ie. when drawing charts by hand), and when reproducing older range-based calculations.


## Related classes

<project:#mqr.spc.XBarParams>, <project:#mqr.spc.RParams>, <project:#mqr.spc.SParams>.


## References

[^1]:   Mahmoud, M. A., Henderson, G. R., Epprecht, E. K., & Woodall, W. H. (2010).
        Estimating the standard deviation in quality-control applications.
        Journal of Quality Technology, 42(4), 348-357.
