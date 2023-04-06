import h5py
import pandas as pd
import numpy as np
# import math

# Create the HDF5 file
with h5py.File('data.hdf5', 'w') as f:

    # Create the dataset group
    dataset = f.create_group("dataset")

    # Create the groups for training and testing
    dataset.create_group("train")
    dataset.create_group("test")

    # Create groups for each memeber's data
    kevin = f.create_group("kevin")
    jacob = f.create_group("jacob")
    taylor = f.create_group("taylor")

    # Concatenate each member's walking data into one dataset and same for jumping
    def concatenate_data(directory, group):
        jumpBP = pd.read_csv(directory + "0.csv").to_numpy()
        jumpFP = pd.read_csv(directory + "1.csv").to_numpy()
        jumpJP = pd.read_csv(directory + "2.csv").to_numpy()
        jumpH = pd.read_csv(directory + "3.csv").to_numpy()

        walkBP = pd.read_csv(directory + "4.csv").to_numpy()
        walkFP = pd.read_csv(directory + "5.csv").to_numpy()
        walkJP = pd.read_csv(directory + "6.csv").to_numpy()
        walkH = pd.read_csv(directory + "7.csv").to_numpy()

        # jumpArray = [jumpBP, jumpFP, jumpJP, jumpH]
        # walkArray = [walkBP, walkFP, walkJP, walkH]

        # jumpSegments = []
        # walkSegments = []

        # # print(jumpBP[len(jumpBP) - 1][0])

        # for i in range(0, len(jumpArray)): # go through each array in JumpArray
        #     fiveSecondSegmentCount = math.floor(jumpArray[i][len(jumpArray[i]) - 1][0] / 5)
        #     for j in range(fiveSecondSegmentCount): # go through each 5sec window in current array
        #         five = (jumpArray[i][:, 0] >= 0 + j*5) & (jumpArray[i][:, 0] <= 5 + j*5) #Indexes the time column based on 5 second intervals
        #         jumpSegments.append(jumpArray[i][five])


        # for i in range(0, len(walkArray)): # go through each array in JumpArray
        #     fiveSecondSegmentCount = math.floor(walkArray[i][len(walkArray[i]) - 1][0] / 5)
        #     for j in range(fiveSecondSegmentCount): # go through each 5sec window in current array
        #         five = (walkArray[i][:, 0] >= 0 + j*5) & (walkArray[i][:, 0] <= 5 + j*5) #Indexes the time column for each walk array (eg. walkBP, walkFP, ... etc, based on 5 second intervals
        #         walkSegments.append(walkArray[i][five])
            
        # # print(jumpSegments)
        # print(walkSegments)


        # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, random_state=0)
        # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, random_state=0)


        # Combines the jumping data into one big array as well as the walking data
        jumping = np.concatenate((jumpBP, jumpFP, jumpJP, jumpH), axis=0)
        walking = np.concatenate((walkBP, walkFP, walkJP, walkH), axis=0)

        group.create_dataset("jumping", data=jumping)
        group.create_dataset("walking", data=walking)



        

    concatenate_data("./jacob/", jacob)
    concatenate_data("./taylor/", taylor)
    concatenate_data("./kevin/", kevin)



    # mask = (jump.iloc[:, 0] >= 0) & (jump.iloc[:, 0] <= 5)









# TEST CODE TO SEE IF HDF Creation worked
# Open the HDF5 file in read-only mode
# with h5py.File('./data.hdf5', 'r') as f:
#     # Print the keys of the root group to see what datasets are available
#     print(list(f.keys()))

#     # Access a subgroup by its name
#     subgroup = f['dataset']

#     print(list(subgroup.keys()))
    # Access a dataset by its name
    # dataset = subgroup['']

    # Read the data from the dataset into a NumPy array
    # data = dataset[()]

#     # Print the shape of the data array
    # print(data.shape)

