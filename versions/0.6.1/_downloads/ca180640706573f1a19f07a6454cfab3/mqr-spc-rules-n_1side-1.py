data = pd.DataFrame({
    'Xbar': np.array([9, 10, 11, 13, 12, 8])
#                            ^^^^^^^^^^ (3 > target)
})

params = mqr.spc.XBarParams(centre=10, sigma=1)
stat = params.statistic(data)
rule = mqr.spc.rules.n_1side(3)

fig, ax = plt.subplots(figsize=(7, 3))
mqr.plot.spc.chart(stat, params, ax=ax)
mqr.plot.spc.alarms(stat, params, rule, ax=ax)