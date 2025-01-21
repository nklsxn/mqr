from statsmodels.formula.api import ols

data = pd.read_csv(mqr.sample_data('anova-glue.csv'), index_col='Run')
model = ols('adhesion_force ~ C(primer) + C(glue)', data=data)
result = model.fit()

fig, axs = plt.subplots(1, 2, figsize=(4, 2), layout='constrained')
mqr.plot.tools.sharey(fig, axs)
mqr.plot.anova.model_means(
    result,
    response='adhesion_force',
    factors=['primer', 'glue'],
    axs=axs)