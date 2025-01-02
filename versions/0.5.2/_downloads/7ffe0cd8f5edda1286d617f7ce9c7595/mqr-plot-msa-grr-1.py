columns = ['RUNID', 'WAFERID', 'PROBE', 'MONTH', 'DAY', 'OPERATOR', 'TEMP', 'AVERAGE', 'STDDEV',]
dtype = {
    'WAFERID': int,
    'PROBE':int,
}

data = pd.read_csv(
    'https://www.itl.nist.gov/div898/software/dataplot/data/MPC61.DAT',
    skiprows=50,
    header=None,
    names=columns,
    sep='\\s+',
    dtype=dtype,
)
data['REPEAT'] = np.repeat([1,2,3,4,5,6,7,8,9,10,11,12], 25)