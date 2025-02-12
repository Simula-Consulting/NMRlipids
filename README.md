# Tutorial showing how a neural net can be trained to predict total densities from form factors with the NMR lipids database

## Installing dependencies
The repository uses Poetry for package handling. Dependencies for running notebooks can be installed using:
\```bash
poetry install
\```

## About this repository
The material in this repository is intended to give a starting point to the [NMRlipids databank](https://nmrlipids.github.io).

This includes an overview of accessing data using the NMR lipids API, exploring the data, and examples of using machine learning to make predictions of total electron densities from form factors. 

The notebook contains the following steps:
1) Download the data using the [NMR lipids API](https://nmrlipids.github.io/databankLibrary.html).  
2) Explore the data, and preprocess the data for the machine learning pipeline.  
3) Split data into train and test sets.
4) Implement different neural networks and perform hyperparameter tuning
5) Train and evaluate the performance of the neural networks. 

## Background of the repository.
This repository is based on initial work by the [CellScatter project](https://github.com/K123AsJ0k1/CellScatter/tree/main). 
