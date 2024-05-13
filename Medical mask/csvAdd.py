import pandas as pd

# Load the csv file
df = pd.read_csv('MyTrain.csv')

# Define the list of mask classes
mask_classes = ['mask_colorful', 'mask_surgical', 'face_with_mask', 'face_with_mask_incorrect', 'gas_mask']

# Create a new column 'maskValue' based on the 'classname' column
df['maskValue'] = df['classname'].apply(lambda x: 'face_with_mask' if x in mask_classes else 'face_no_mask')

# Save the modified csv file
df.to_csv('MyTrain.csv', index=False)