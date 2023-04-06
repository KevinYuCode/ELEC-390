import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

dataset_name = "____"
dataset = pd.read_csv(dataset_name)
df = dataset.iloc[:, 1:-1]
labels = dataset.iloc[-1:0, -1]


# Missing data
def fill(df):
    # change all ‘-’s to NaN
    print(df.isna().sum())  # check NaNs in dataset
    print(np.where(df.isna()))
    print(df.isna().sum())
    print(np.where(df == '-'))
    print((df == '-').sum())
    df.mask(df == '-', other=np.nan, inplace=True)
    df = df.astype('float64')
    df.interpolate(method='linear', inplace=True)  # Linear Regression/interpolation imputation


def filtering(df):
    # setting the frequency of sampling and the frequency of sine
    freq_sampling = 900
    freq = 5
    # how many samples do we need
    n_sample = 700
    x_input = np.arange(n_sample)
    # building the sine function
    y = np.sin((2 * np.pi * freq * x_input) / freq_sampling)
    # converting y to dataframe
    y_df = pd.DataFrame(y)
    # applying SMA
    window_size = 21
    y_sma = y_df.rolling(window_size).mean()
    # plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x_input, y, linewidth=5)
    ax.plot(x_input, y_df)
    ax.plot(x_input, y_sma.to_numpy(), linewidth=5)
    ax.legend(['sine without noise', 'noisy sine', 'SMA 20'])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()


def normalization(df):
    # normalizing
    sc = preprocessing.StandardScaler()
    df = sc.fit_transform(df)


