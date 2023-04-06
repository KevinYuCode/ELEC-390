import pandas as pd
import numpy as np
import h5py
from scipy import stats
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model

data = h5py.File("data.hdf5", 'r')
jumping_df = pd.DataFrame()
walking_df = pd.DataFrame()

for name, group in data.items():
    if isinstance(group, h5py.Group):  # Checks if there is a valid group in the HDF5 file
        for name, dataset in group.items():
            if "jumping" in name:
                jumping_df = pd.concat(
                    [jumping_df, pd.DataFrame(dataset)], axis=0, ignore_index=True)
            if "walking" in name:
                walking_df = pd.concat(
                    [walking_df, pd.DataFrame(dataset)], axis=0, ignore_index=True)

jumping_df['label'] = "jumping"
walking_df['label'] = "walking"

jumping_df.columns = ['time', 'x-acceleration', 'y-acceleration', 'z-acceleration', 'absolute-acceleration', 'label']
walking_df.columns = ['time', 'x-acceleration', 'y-acceleration', 'z-acceleration', 'absolute-acceleration', 'label']

jumping_df.drop(["time", "absolute-acceleration"], axis=1, inplace=True)
walking_df.drop(["time", "absolute-acceleration"], axis=1, inplace=True)

print(jumping_df.head())

def featureExtraction(dataframe):
    # setting window size.
    window_size = 500

    features = pd.DataFrame(columns=['max_x', 'max_y', 'max_z', 'min_x', 'min_y', 'min_z', 'range_x', 'range_y',\
                                      'range_z', 'mean_x','mean_y','mean_z', 'median_x', 'median_y', 'median_z',\
                                          'variance_x', 'variance_y', 'variance_z', 'skew_x', 'skew_y', 'skew_z',\
                                              'std_x', 'std_y', 'std_z', 'kurtosis_x','kurtosis_y','kurtosis_z'\
                                                 ])#, 'slope_x','slope_y','slope_z'])
    
    df_split = np.split(dataframe, range(window_size, dataframe.shape[0], window_size))

    # Calculate all features
    for i in range(0, len(df_split)):
        # Max
        features.loc[i, 'max_x'] = df_split[i].max()['x-acceleration'] 
        features.loc[i, 'max_y'] = df_split[i].max()['y-acceleration'] 
        features.loc[i, 'max_z'] = df_split[i].max()['z-acceleration'] 

        # Minimum
        features.loc[i, 'min_x'] = df_split[i].min()['x-acceleration'] 
        features.loc[i, 'min_y'] = df_split[i].min()['y-acceleration'] 
        features.loc[i, 'min_z'] = df_split[i].min()['z-acceleration'] 

        # Range
        features.loc[i, 'range_x'] = np.ptp(df_split[i]['x-acceleration'], axis=0)
        features.loc[i, 'range_y'] = np.ptp(df_split[i]['y-acceleration'], axis=0)
        features.loc[i, 'range_z'] = np.ptp(df_split[i]['z-acceleration'], axis=0)

        # Mean
        features.loc[i, 'mean_x'] = np.mean(df_split[i]['x-acceleration'], axis=0)
        features.loc[i, 'mean_y'] = np.mean(df_split[i]['y-acceleration'], axis=0)
        features.loc[i, 'mean_z'] = np.mean(df_split[i]['z-acceleration'], axis=0)

        # Median
        features.loc[i, 'median_x'] = np.median(df_split[i]['x-acceleration'], axis=0)
        features.loc[i, 'median_y'] = np.median(df_split[i]['y-acceleration'], axis=0)
        features.loc[i, 'median_z'] = np.median(df_split[i]['z-acceleration'], axis=0)

        # Variance
        features.loc[i, 'variance_x'] = np.var(df_split[i]['x-acceleration'], axis=0)
        features.loc[i, 'variance_y'] = np.var(df_split[i]['y-acceleration'], axis=0)
        features.loc[i, 'variance_z'] = np.var(df_split[i]['z-acceleration'], axis=0)

        # Skewness
        features.loc[i, 'skewness_x'] = stats.skew(df_split[i]['x-acceleration'], axis=0)
        features.loc[i, 'skewness_y'] = stats.skew(df_split[i]['y-acceleration'], axis=0)
        features.loc[i, 'skewness_z'] = stats.skew(df_split[i]['z-acceleration'], axis=0)

        # Std
        features.loc[i, 'std_x'] = np.sqrt(np.std(df_split[i]['x-acceleration'] ** 2))
        features.loc[i, 'std_y'] = np.sqrt(np.std(df_split[i]['y-acceleration'] ** 2))
        features.loc[i, 'std_z'] = np.sqrt(np.std(df_split[i]['z-acceleration'] ** 2))

        # Kurtosis
        features.loc[i, 'kurtosis_x'] = stats.kurtosis(df_split[i]['x-acceleration'], axis=0)
        features.loc[i, 'kurtosis_y'] = stats.kurtosis(df_split[i]['y-acceleration'], axis=0)
        features.loc[i, 'kurtosis_z'] = stats.kurtosis(df_split[i]['z-acceleration'], axis=0)

        # Slope
        features.loc[i, 'slope_x'] = np.polyfit(range(window_size), df_split[i]['x-acceleration'], 1)[0]
        features.loc[i, 'slope_y'] = np.polyfit(range(window_size), df_split[i]['y-acceleration'], 1)[0]
        features.loc[i, 'slope_z'] = np.polyfit(range(window_size), df_split[i]['z-acceleration'], 1)[0]

    return features

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


# def filtering(df):
#     # setting the frequency of sampling and the frequency of sine
#     freq_sampling = 900
#     freq = 5
#     # how many samples do we need
#     n_sample = 700
#     x_input = np.arange(n_sample)
#     # building the sine function
#     y = np.sin((2 * np.pi * freq * x_input) / freq_sampling)
#     # converting y to dataframe
#     y_df = pd.DataFrame(y)
#     # applying SMA
#     window_size = 21
#     y_sma = y_df.rolling(window_size).mean()
#     # plotting
#     fig, ax = plt.subplots(figsize=(10, 10))
#     ax.plot(x_input, y, linewidth=5)
#     ax.plot(x_input, y_df)
#     ax.plot(x_input, y_sma.to_numpy(), linewidth=5)
#     ax.legend(['sine without noise', 'noisy sine', 'SMA 20'])
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     plt.show()


jumping_features = featureExtraction(jumping_df)
walking_features = featureExtraction(walking_df)
features_df = pd.concat([jumping_features, walking_features], axis=0, ignore_index=True)

jumpingLabels = np.zeros((jumping_features.shape[0], 1))
walkingLabels = np.ones((walking_features.shape[0], 1))
labels = np.concatenate((jumpingLabels, walkingLabels), axis=0)


# features_df.mask(df == '-', other=np.nan, inplace=True)
# features_df = features_df.astype('float64')
# features_df.interpolate(method='linear', inplace=True)

# print(features_df)
# features = preprocessing.normalize(features_df)

features_df = fill(features_df)
features_df = preprocessing.normalize(features_df)#.fit_transform(features_df)

# splitting the data into training and testing data
x_train, x_test, y_train, y_test = model_selection.train_test_split(features_df, labels, test_size=0.1, random_state=0)

# training the model
model = linear_model.LogisticRegression(random_state=0).fit(x_train, np.ravel(y_train))

# testing the model
print("Accuracy of the model is: ", model.score(x_test, y_test))
