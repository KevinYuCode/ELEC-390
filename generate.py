import h5py
import pandas as pd
import numpy as np

def generate():

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

            # Combines the jumping data into one big array as well as the walking data
            jumping = np.concatenate((jumpBP, jumpFP, jumpJP, jumpH), axis=0)
            walking = np.concatenate((walkBP, walkFP, walkJP, walkH), axis=0)

            group.create_dataset("jumping", data=jumping)
            group.create_dataset("walking", data=walking)



            

        concatenate_data("./jacob/", jacob)
        concatenate_data("./taylor/", taylor)
        concatenate_data("./kevin/", kevin)



generate()







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

