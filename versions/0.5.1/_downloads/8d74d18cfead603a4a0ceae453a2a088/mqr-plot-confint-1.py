import mqr

ci = mqr.inference.mean.confint_1sample(
    np.array([1, 5, 4, 3, 6, 2, 3, 3, 2, 5]),
    conf=0.98,
)

fig, ax = plt.subplots(figsize=(5, 2))
mqr.plot.confint(ci, ax, hyp_value=3)
plt.show()