# Higgs Boson Signal and Background Classification using XGBoost
## This Python program is designed to classify signal and background events in the context of Higgs boson searches using the XGBoost machine learning algorithm. It operates on particle physics data, specifically focusing on differentiating between events where a Higgs boson is produced (signal) and other processes (background). The program utilizes a variety of Python libraries for data manipulation, machine learning, and visualization.

## main code:
> python XGBoost_training_mask_mass.py
## Key Features:
Data Preprocessing: Utilizes uproot to load data from ROOT files, a common format in particle physics. Data is further processed using pandas for dataframe manipulation and awkward for handling jagged arrays.

Feature Selection: put 9 input features based on Hgg group recommendation

Model Training and Evaluation: Leverages XGBoost for training gradient boosting models, with hyperparameter tuning via bayesain optimization. Evaluate based on the saved model with model.predict_proba to get the transformed score(0-1)

Weighting and Reweighting: Implements complex weighting schemes to account for discrepancies in signal and background efficiencies, including mass window selection and specific weight adjustments like abs weight, reweight abs weight, and W_sig.

Visualization: Generates ROC curves and signal-background discrimination plots using matplotlib to evaluate model performance.

Usage Instructions:
Data Setup: Ensure input ROOT files containing signal and background events are correctly placed in the specified directories.

Configuration: Modify hyperparameters and input features as needed. The train_model function allows for customization of the learning rate, depth, estimators, and more.

Model Training: Run the train_model function with the desired dataset and hyperparameters. This function will train the model and save it to disk.

Evaluation: After training, evaluate the model's performance using ROC curves and distribution plots generated by the program.
## Example:
To train a model on the UL17 dataset with specific hyperparameters:
 python
Copy code
> train_model(events_sig=events_sig_17, events_bkg=events_bkg_17, colsample_bytree_v=0.5, gamma_v=0.308, learning_rate_v=0.0083, max_depth_v=4, min_child_weight_v=10, n_estimators_v=7981, reg_alpha_v=20, reg_lambda_v=10.9, subsample_v=0.5, is_mass_window=False, weight_style='weight_absRatio', samples="UL17_custom", input_features=["leadmva", "subleadmva", "leadptom", "subleadptom", "leadeta", "subleadeta", "CosPhi", "vtxprob", "sigmarv", "sigmawv"]) 


## Dependencies:
xgboost: For model training and evaluation.

sklearn: For model selection and metric computation.

pandas, numpy: For data manipulation.

matplotlib: For plotting.

uproot, awkward: For reading ROOT files and handling particle physics data structures.

pickle: For model serialization.

## Notes:
Ensure that the working directory contains the necessary ROOT files and that the output directories for saving models and plots are correctly specified.
The program includes options for feature selection and hyperparameter tuning, which can significantly impact model performance.
The choice of weights and mass window selection is crucial for the physical significance of the model's predictions.