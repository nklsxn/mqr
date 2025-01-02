fig, ax = plt.subplots(figsize=(7, 3))

# Raw data
data = pd.read_csv(mqr.sample_data('study-random-5x5.csv'))

# Prepare data
study = mqr.summary.Study(data, measurements=['KPO1'])
specs = {
    'KPO1': mqr.process.Specification(160, 150, 170),
}
process = mqr.process.Process(study, specs)

# Plot capability
mqr.plot.process.capability(process, 'KPO1', ax)