data = pd.DataFrame({
    'Xbar': np.array([11, 9, 12, 10, 9, 8, 12])
#                            ^^^^^^^^^^^^ (4 trending)
})

params = mqr.spc.XBarParams(centre=10, sigma=1)
stat = params.statistic(data)
rule = mqr.spc.rules.n_trending(4)

fig, ax = plt.subplots(figsize=(7, 3))
mqr.plot.spc.chart(stat, params, ax=ax)
mqr.plot.spc.alarms(stat, params, rule, ax=ax)