data = pd.read_csv(mqr.sample_data('anova-glue.csv'), index_col='Run')

fig, axs = plt.subplots(figsize=(3, 2), layout='constrained')
mqr.plot.anova.interactions(
    data,
    response='adhesion_force',
    group='primer',
    factors=['glue'],
    axs=axs)