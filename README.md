# Information about the project

Preprocess_all = 
Converts Raw Oasis data into usable format
1. Reads raw brain scans from Oasis
2. Performs skull-stripping, removing non-brain tissue
3. Resizes to 96x96x96
4. Normalises intensity values 
5. Saves as .nii.gz file

-> processed data foes to data/processed

Train.py = 
Trains one model
1. Loads training and validation data
2. Creates ResNet3D
3. Trains for given epochs
4. Saves model 
5. Prints training progress

Evaluate.py =
Tests trained model on test set
1. Loads trained model from saved model
2. Runs predictions on test set
3. Calculates metrics
4. Generates confusion matrix and ROC curve
5. Saves figures to results folder

Dataset_Summary.py = 
Analyses patient demographics
1. Loads metadata.csv
2. Counts patients per diagnosis
3. Calculates age/sex/education statistics
4. Creates demographic plots

Dataset.py = 
Loads data during training
1. PyTorch Dataset class: reads scan files
2. Applies augmentation during training
3. Returns (image, label) pairs to the model
4. No output as it is a helper class

Resnet3d.py =
Defines the neural network structure
1. BasicBlock3D - building block (2 conv layers + skip connection)
2. ResNet3D - full network (4 layer groups)
3. ResNet3D_18 - small version (2,2,2,2 blocks)
4. Used by running train.py as it is a defining model

