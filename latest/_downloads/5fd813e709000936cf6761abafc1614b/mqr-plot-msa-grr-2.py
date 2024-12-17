tol = 2*8.0
names = mqr.msa.NameMapping(
    part='WAFERID',
    operator='PROBE',
    measurement='AVERAGE')
grr = mqr.msa.GRR(
    data.query('RUNID==1'),
    tolerance=tol,
    names=names,
    include_interaction=True)

fig, axs = plt.subplots(3, 2, figsize=(10, 6), layout='constrained')
mqr.plot.msa.grr(grr, axs=axs)