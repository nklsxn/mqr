data = pd.read_csv(mqr.sample_data('anova-glue.csv'), index_col='Run')

fig, axs = plt.subplots(1, 2, figsize=(5, 2), layout='constrained')
mqr.plot.tools.sharey(fig, axs)
mqr.plot.anova.main_effects(
    data,
    response='adhesion_force',
    factors=['primer', 'glue'],
    axs=axs)