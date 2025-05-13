fig, ax = plt.subplots(figsize=(7, 3))

# Make the new rule
combine_rule = mqr.spc.rules.combine(
    np.logical_or,
    mqr.spc.rules.limits(),
    mqr.spc.rules.aofb_nsigma(a=3, b=4, n=2))

# Create violating data for demonstration
mean = 5
sigma = 1
zscore = np.array([[0, 3.5, 0, 0, 0, 2.5, 1, 2.5, 2.5, 0]]).T
#                      ^^^(>UCL)     ^^^^^^^^^^^^^^^^(3/4>2)
data = pd.DataFrame(mean + zscore * sigma)

# Create parameters and calculate statistic for the example chart
params = mqr.spc.XBarParams(centre=mean, sigma=sigma)
stat = params.statistic(data)

# Show the chart and alarms overlay for the combined rule
mqr.plot.spc.chart(stat, params, ax=ax)
mqr.plot.spc.alarms(stat, params, combine_rule, ax=ax)