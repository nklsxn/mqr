fig, ax = plt.subplots(figsize=(7, 3))

# Raw data
np.random.seed(0)
x = pd.DataFrame(
    scipy.stats.norm(1, 5).rvs([20, 6]),
    columns=range(6))

# Parameters
params = mqr.spc.XBarParams(centre=1, sigma=5, nsigma=2)
stat = params.statistic(x)
rule = mqr.spc.rules.limits()

# Charts
mqr.plot.spc.chart(stat, params, ax=ax)
mqr.plot.spc.alarms(stat, params, rule, ax=ax)