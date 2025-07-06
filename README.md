# End-to-End-Dynamic-Price-Optimization

## Overview
The Dynamic Price Optimization Model is designed to predict the optimal price of products using various input features such as Surge Index, Competitor pricing, product characteristics, and historical pricing data. By using the XGBoost regression model, this solution aims to maximize pricing efficiency while ensuring competitive market prices.

This project is built with Python, and the end-to-end pipeline involves data preprocessing, model training, and deployment using Azure Machine Learning and MLflow.

## Project Architecture
The project is structured in the following main components:

- Data Collection and Preprocessing: Raw data is collected and preprocessed to prepare it for training the machine learning model.
- Model Training: The preprocessed data is fed into an XGBoost regression model to predict optimal pricing.
- Model Deployment: The trained model is deployed using Azure Machine Learning as Real Time Endpoint.
- Model Tracking with MLflow: We use MLflow to track model parameters, metrics, and the deployment process, ensuring reproducibility and version control.

## Technologies Used
- Python: Programming language used for data processing and model development.
- XGBoost: An efficient gradient boosting library used for regression tasks.
- Azure Machine Learning: Cloud-based platform for deploying and managing models.
- MLflow: Open-source platform to manage the machine learning lifecycle, including experimentation, reproducibility, and deployment.
- Pandas & NumPy: Libraries for data manipulation and analysis.
- Scikit-learn: Library for data preprocessing, model evaluation, and splitting data.

## Steps to Train and Deploy the model

### Clone the Repository:
```
git clone https://github.com/your-username/End-to-End-Dynamic-Price-Optimization.git
cd End-to-End-Dynamic-Price-Optimization
```

Install Dependencies:

Create the Conda Environment:
Save the above content into a file named model-env.yaml and then run the following command to create the environment:
```
conda env create -f model-env.yaml
```
Activate the Environment:

Once the environment is created, activate it using:
```
conda activate model-env
```
Run the Script:
Run the train.py script with the necessary arguments:
```
python train.py --training_data path_to_your_data.csv --reg_rate 0.01
```

Deploy on Azure:
Use Azure Machine Learning services to deploy the model in production for real-time predictions.
