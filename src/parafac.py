import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorly.decomposition import parafac
from sklearn.preprocessing import LabelEncoder
import tensorly as ts

def compute_parafac(df):
    
    df = df[df.groupby('ID')['temps_inj'].transform('nunique') == 4]
    new = df.set_index(['ID', 'temps_inj'])
    wide_df = new.unstack(level='temps_inj')
    X = wide_df.drop(['classe_name'],axis=1)
    y = wide_df['classe_name']['ART']

    N = len(y)
    n2 = 4
    X_array = np.array(X).reshape(N,-1,n2)
    y_encoded = LabelEncoder().fit_transform(y)

    t = ts.tensor(X_array)
    weights, factors = parafac(t, rank=2, normalize_factors=True)

    plt.figure()
    plt.plot(factors[2])
    plt.figure()
    plt.scatter(*zip(*factors[0]), c=y_encoded)