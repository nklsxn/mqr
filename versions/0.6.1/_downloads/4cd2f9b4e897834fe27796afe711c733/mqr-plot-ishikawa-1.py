import mqr

fig, ax = plt.subplots(figsize=(5, 3))

problem = 'Problem'
causes = {
    'Material':    ['Cause1a', 'Cause1b', 'Cause1c'],
    'People':      ['Cause2a', 'Cause2b'],
    'Machine':     ['Cause3a', 'Cause3b', 'Cause3c'],
    'Measurement': ['Cause4a', 'Cause4b', 'Cause4c', 'Cause4d'],
    'Method':      ['Cause5a', 'Cause5b', 'Cause5c', 'Cause5c'],
    'Environment': ['Cause6a', 'Cause6b', 'Cause6c'],
}
mqr.plot.ishikawa(problem, causes, ax=ax)

plt.show()