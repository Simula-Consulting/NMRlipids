# Tutorial showing how a neural net can be trained to predict total densities from form factors with the NMR lipids database

The notebook step by step: 
* Initialize the NMR lipids databank
* Gather all available form factors and total densities in data frames
* Preprocess the data to deal with different dimensions due to different experimental setups
* Split data into a train and a test set
* Train and evaluate a neural net that predicts total densities from form factors
