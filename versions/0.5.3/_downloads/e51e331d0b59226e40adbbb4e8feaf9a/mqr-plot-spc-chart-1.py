fig, ax = plt.subplots(figsize=(7, 3))

# Raw data
np.random.seed(0)
x = pd.DataFrame(scipy.stats.norm(1, 5).rvs([20, 6]))

# Parameters
params = mqr.spc.XBarParams(centre=1, sigma=5)
stat = params.statistic(x)

# Charts
mqr.plot.spc.chart(stat, params, ax=ax)