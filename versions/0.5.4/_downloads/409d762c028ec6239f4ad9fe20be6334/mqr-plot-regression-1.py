import statsmodels

# Raw data
data = pd.read_csv(mqr.sample_data('anova-glue.csv'), index_col='Run')

# Fit a linear model
model = statsmodels.formula.api.ols('adhesion_force ~ C(primer) * C(glue)', data)
result = model.fit()