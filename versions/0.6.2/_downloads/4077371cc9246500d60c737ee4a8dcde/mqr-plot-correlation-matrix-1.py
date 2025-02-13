data = pd.read_csv(mqr.sample_data('study-random-5x5.csv'))

fig, ax = plt.subplots(5, 5, figsize=(6, 6))
mqr.plot.correlation.matrix(
    data.loc[:, 'KPI1':'KPO2'],
    show_conf=True,
    ax=ax,
    fig=fig)
plt.show()    