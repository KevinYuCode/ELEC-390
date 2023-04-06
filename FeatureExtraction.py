import pandas as pd
import matplotlib.pyplot as plt

"""Feature Extraction"""
dataset_name = "____"
data = pd.read_csv(dataset_name, on_bad_lines='skip', header=None)
fig, ax = plt.subplots()
data.iloc[:, -1].plot(ax=ax, linewidth=3)
ax.set_title('Sampling Frequency: 125Hz')
ax.set_xlabel('data points')
ax.set_ylabel('Normalized value')
plt.show()
# extracting mean , std , max , kurtosis , and skewness for one of the samples
# if you need to do the feature extraction for all the samples, you have to use a for loop creating an empty data frame
features = pd.DataFrame(columns=['max', 'min', 'range', 'mean', 'median', 'variance', 'skew', 'std', 'kurtosis', 'slope'])
# setting window size.
window_size = 31
# feature extraction using the notion of rolling window
features['max'] = data.iloc[:, -1].rolling(window=window_size).max()
features['min'] = data.iloc[:, -1].rolling(window=window_size).min()
features['range'] = data.iloc[:, -1].rolling(window=window_size).range()
features['mean'] = data.iloc[:, -1].rolling(window=window_size).mean()
features['median'] = data.iloc[:, -1].rolling(window=window_size).median()
features['variance'] = data.iloc[:, -1].rolling(window=window_size).variance()
features['skew'] = data.iloc[:, -1].rolling(window=window_size).skew()
features['std'] = data.iloc[:, -1].rolling(window=window_size).std()
features['kurtosis'] = data.iloc[:, -1].rolling(window=window_size).kurt()
features['slope'] = data.iloc[:, -1].rolling(window=window_size).apply(lambda x: np.polyfit(range(window_size), x, 1)[0])


features = features.dropna()
print(features)
