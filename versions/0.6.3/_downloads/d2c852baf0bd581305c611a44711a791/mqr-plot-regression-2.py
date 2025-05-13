fig, axs = plt.subplots(3, 2, figsize=(8, 5), layout='constrained')

# show the four residuals plots
mqr.plot.regression.residuals(result.resid, result.fittedvalues, axs=axs[:2, :])

# plot residuals against each factor
mqr.plot.regression.res_v_factor(result.resid, data['primer'], axs[2, 0])
mqr.plot.regression.res_v_factor(result.resid, data['glue'], axs[2, 1])

# show Cook's Distance measure of influence.
mqr.plot.regression.influence(result, 'cooks_dist', axs[1, 0])