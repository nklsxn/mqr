data = pd.DataFrame({
    'Xbar': np.array([11, 13, 10, 12, 12, 12, 9, 11, 12])
#                         ^^^^^^^^^^^^^^^^^^ (4/5>=2sigma)
})

params = mqr.spc.XBarParams(centre=10, sigma=1)
stat = params.statistic(data)
rule = mqr.spc.rules.aofb_nsigma(a=4, b=5, n=2)

fig, ax = plt.subplots(figsize=(7, 3))
mqr.plot.spc.chart(stat, params, ax=ax)
mqr.plot.spc.alarms(stat, params, rule, ax=ax)