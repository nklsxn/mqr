data = pd.DataFrame({
    'Xbar': np.array([9, 10, 9, 13, 12, 10])
#                               ^^ (>UCL)
})

params = mqr.spc.XBarParams(centre=10, sigma=1)
stat = params.statistic(data)
rule = mqr.spc.rules.limits()

fig, ax = plt.subplots(figsize=(7, 3))
mqr.plot.spc.chart(stat, params, ax=ax)
mqr.plot.spc.alarms(stat, params, rule, ax=ax)