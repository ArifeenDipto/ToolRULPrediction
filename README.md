# ToolRULPrediction
This repository contains python codes for processing, feature extractions, feature selection, feature fusion and RUL prediction models for CNC machine tools.

# Dataset
The dataset used in this experiment comes from the paper titled: **Multivariate time series data of milling processes with varying tool wear and machine tools**

Paper link - https://www.sciencedirect.com/science/article/pii/S2352340923006741
# File Descriptions:

**CSVFileGeneration.py**
The raw data from the paper comes in H5 format. The H5 files first need to be converted into CSV files for data driven modelling.
The python **h5py** library is needed to be installed for conversion.

This file shows how to convert the H5 files to CSV files for time series analysis.
