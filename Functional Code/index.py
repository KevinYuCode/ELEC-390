import h5py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math as math
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import iqr
from sklearn.linear_model import LogisticRegression


# ------------------------------------------ CREATING THE HDF5 FILE (NO STORAGE YET) ------------------------------------------ #

f = h5py.File('data.hdf5', 'w')

# Create the dataset group
dataset = f.create_group("dataset")

# Create the groups for training and testing
train = dataset.create_group("train")
test = dataset.create_group("test")

# Create groups for each memeber's data
kevin = f.create_group("kevin")
jacob = f.create_group("jacob")
taylor = f.create_group("taylor")

# Concatenate each member's walking data into one dataset and same for jumping
def get_team_member_data(directory, group):

    # Read all of the csv data for jumping
    jumpBP = pd.read_csv(directory + "0.csv")
    jumpFP = pd.read_csv(directory + "1.csv")
    jumpJP = pd.read_csv(directory + "2.csv")
    jumpH = pd.read_csv(directory + "3.csv")

    # Read all of the walking data for walking
    walkBP = pd.read_csv(directory + "4.csv")
    walkFP = pd.read_csv(directory + "5.csv")
    walkJP = pd.read_csv(directory + "6.csv")
    walkH = pd.read_csv(directory + "7.csv")

    jumping = pd.concat([jumpBP, jumpFP, jumpJP, jumpH], axis=0) # Combine all jumping data for each phone position
    walking = pd.concat([walkBP, walkFP, walkJP, walkH], axis=0) # Combine all walking data for each phone position

    jumping.drop("Absolute acceleration (m/s^2)", axis=1, inplace=True)
    walking.drop("Absolute acceleration (m/s^2)", axis=1, inplace=True)

    # Stores the collective jumping and walking data for each memeber and creates a dataset under there name
    group.create_dataset("jumping", data=jumping)
    group.create_dataset("walking", data=walking)
    return [jumping, walking]

# Get each team members walking and jumping data
jacobs_data = get_team_member_data("../jacob/", jacob)
taylors_data = get_team_member_data("../taylor/", taylor)
kevins_data = get_team_member_data("../kevin/", kevin)

walking_data = pd.concat([jacobs_data[1], taylors_data[1], kevins_data[1]]) # Combine all walking data for each team member
jumping_data = pd.concat([jacobs_data[0], taylors_data[0], kevins_data[0]]) # Combine all jumping data for each team member

# #Verify that the HDF5 file works
# # print(f.keys()) #Prints the root directory
# # print(f["dataset"].keys()) #Prints the root directory
# # print(pd.DataFrame((f['kevin']['jumping'])).head()) #prints out data in team members group
# # print(pd.DataFrame((f['kevin']['walking'])).head()) #prints out data in team members group

# ------------------------------------------ PREPROCESSING ------------------------------------------ #

# Apply a moving average filter with a window size of 3
walking_data = walking_data.rolling(window=10).mean()
jumping_data = jumping_data.rolling(window=10).mean()

# Remove the NaN values from the walking and jumping data
walking_data.dropna(inplace=True)
jumping_data.dropna(inplace=True)

# ------------------------------------------ FEATURE EXTRACTION ------------------------------------------ #
# Split the data into 5 second segements (since its 100hz frequency we know that 100 samples are recorded in 1 sec therefore 500 samples in 5 seconds)
split_walking_data = np.array_split(walking_data, len(walking_data)/500, axis=0)
split_jumping_data = np.array_split(jumping_data, len(jumping_data)/500, axis=0)

def get_features(split_data_frame):
    features = pd.DataFrame(columns=['min_x', 'min_y', 'min_z', 
                                      'max_x', 'max_y', 'max_z', 
                                      'mean_x', 'mean_y', 'mean_z', 
                                      'range_x', 'range_y', 'range_z', 
                                      'median_x', 'median_y', 'median_z',
                                      'variance_x','variance_y', 'variance_z',  
                                      'skew_x', 'skew_y', 'skew_z', 
                                      'std_x', 'std_y', 'std_z', 
                                      'kurt_x', 'kurt_y', 'kurt_z', 
                                      "iqr_x", "iqr_y", "iqr_z"
                                      ])


    for i in range(0, len(split_data_frame)):
        frameIdx = split_data_frame[i] #The index of the 5 second segment
        
        # Min
        features.loc[i,'min_x'] = frameIdx['Linear Acceleration x (m/s^2)'].min()
        features.loc[i,'min_y'] = frameIdx["Linear Acceleration y (m/s^2)"].min()
        features.loc[i,'min_z'] = frameIdx["Linear Acceleration z (m/s^2)"].min()

        # Max
        features.loc[i,'max_x'] = frameIdx['Linear Acceleration x (m/s^2)'].max()
        features.loc[i,'max_y'] = frameIdx["Linear Acceleration y (m/s^2)"].max()
        features.loc[i,'max_z'] = frameIdx["Linear Acceleration z (m/s^2)"].max()

        # Mean
        features.loc[i, 'mean_x'] = np.mean(frameIdx['Linear Acceleration x (m/s^2)'], axis=0)
        features.loc[i, 'mean_y'] = np.mean(frameIdx["Linear Acceleration y (m/s^2)"], axis=0)
        features.loc[i, 'mean_z'] = np.mean(frameIdx["Linear Acceleration z (m/s^2)"], axis=0)

        # Range
        features.loc[i, 'range_x'] = np.ptp(frameIdx['Linear Acceleration x (m/s^2)'], axis=0)
        features.loc[i, 'range_y'] = np.ptp(frameIdx["Linear Acceleration y (m/s^2)"], axis=0)
        features.loc[i, 'range_z'] = np.ptp(frameIdx["Linear Acceleration z (m/s^2)"], axis=0)

        # Median
        features.loc[i, 'median_x'] = np.median(frameIdx['Linear Acceleration x (m/s^2)'], axis=0)
        features.loc[i, 'median_y'] = np.median(frameIdx["Linear Acceleration y (m/s^2)"], axis=0)
        features.loc[i, 'median_z'] = np.median(frameIdx["Linear Acceleration z (m/s^2)"], axis=0)

        # Variance
        features.loc[i, 'variance_x'] = np.var(frameIdx['Linear Acceleration x (m/s^2)'], axis=0)
        features.loc[i, 'variance_y'] = np.var(frameIdx["Linear Acceleration y (m/s^2)"], axis=0)
        features.loc[i, 'variance_z'] = np.var(frameIdx["Linear Acceleration z (m/s^2)"], axis=0)

        # Skewness
        features.loc[i, 'skew_x'] = stats.skew(frameIdx['Linear Acceleration x (m/s^2)'], axis=0)
        features.loc[i, 'skew_y'] = stats.skew(frameIdx["Linear Acceleration y (m/s^2)"], axis=0)
        features.loc[i, 'skew_z'] = stats.skew(frameIdx["Linear Acceleration z (m/s^2)"], axis=0)

        # Std
        features.loc[i, 'std_x'] = np.sqrt(np.std(frameIdx['Linear Acceleration x (m/s^2)'] ** 2))
        features.loc[i, 'std_y'] = np.sqrt(np.std(frameIdx["Linear Acceleration y (m/s^2)"] ** 2))
        features.loc[i, 'std_z'] = np.sqrt(np.std(frameIdx["Linear Acceleration z (m/s^2)"] ** 2))

        # kurt
        features.loc[i, 'kurt_x'] = stats.kurtosis(frameIdx['Linear Acceleration x (m/s^2)'], axis=0)
        features.loc[i, 'kurt_y'] = stats.kurtosis(frameIdx["Linear Acceleration y (m/s^2)"], axis=0)
        features.loc[i, 'kurt_z'] = stats.kurtosis(frameIdx["Linear Acceleration z (m/s^2)"], axis=0)

        # iqr
        features.loc[i, 'iqr_x'] = iqr(frameIdx['Linear Acceleration x (m/s^2)'])
        features.loc[i, 'iqr_y'] = iqr(frameIdx['Linear Acceleration y (m/s^2)'])
        features.loc[i, 'iqr_z'] = iqr(frameIdx['Linear Acceleration z (m/s^2)'])

    return features

walking_features = get_features(split_walking_data)
jumping_features = get_features(split_jumping_data)

# ------------------------------------------ VISUALIZATION ------------------------------------------ #
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

# Visualizing the walking data for one 5 second segement
ax1.plot(split_walking_data[0].iloc[:, 0], split_walking_data[0].iloc[:, 1:4])
ax1.set_title('Accelerations vs Time (Walking)')
ax1.set_xlabel('Time')
ax1.set_ylabel('Acceleration (m/s)')

# Add a legend
ax1.legend(["X-acceleration", "Y-acceleration", "Z-acceleration"])

# Visualizing the jumping data for one 5 second segement
ax2.plot(split_jumping_data[0].iloc[:, 0], split_jumping_data[0].iloc[:, 1:4])
ax2.set_title('Accelerations vs Time (Jumping)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Acceleration (m/s)')

#Add meta data tags


# Add a legend
ax2.legend(["X-acceleration", "Y-acceleration", "Z-acceleration"])

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.5)

# Display the plot
plt.show()

# ------------------------------------------ REMOVING OUTLIERS ------------------------------------------ #

#Removing outliers
THRESHOLD = 3

# #Get the z-scores for the walking and jumping features
z_scores_walking = (walking_features - walking_features.mean())/ walking_features.std()
z_scores_jumping = (jumping_features - jumping_features.mean())/ jumping_features.std()

# Creates a 2D array of True/False values to determine which point is an outlier in the dataframe
outliers_walking = (abs(z_scores_walking) > THRESHOLD).any(axis=1)
outliers_jumping = (abs(z_scores_jumping) > THRESHOLD).any(axis=1)

# Removes the outlier in the dataframe
walking_features =  walking_features[~outliers_walking]
jumping_features = jumping_features[~outliers_jumping]

# ------------------------------------------ CREATING A CLASSIFIER ------------------------------------------ #
# Store walking and jumping features into one dataframe
all_features = pd.concat([walking_features, jumping_features], axis=0)

# Label the walking features with a 1 and jumping with a 0 and store those values in a np array
walking_labels = np.ones(len(walking_features))
jumping_labels = np.zeros(len(jumping_features))
all_labels = np.concatenate((walking_labels, jumping_labels), axis=0)

# #Normalizing the data
all_features = preprocessing.normalize(all_features)

x_train, x_test, y_train, y_test = \
  train_test_split(all_features, all_labels, test_size=0.1, shuffle=True, random_state=0)

#Creating logistic regiression model object
log_reg = LogisticRegression(random_state=0)

#Fiting the model to the training data
model = log_reg.fit(x_train, y_train)

#Testing the model
print("Model Accuracy: " , model.score(x_test, y_test))

# ------------------------------------------ DATA STORAGE ------------------------------------------ #

#Store the train and test data in the HDF5 file
train.create_dataset('x_train', data = x_train)
train.create_dataset('y_train', data = y_train)
test.create_dataset('x_test', data = x_test)
test.create_dataset('y_test', data = y_test)