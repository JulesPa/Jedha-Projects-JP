In the root of this folder you will find a raw dataset get_around_delay_analysis.xlsx along with the notebook dataexploration.ipynb that explore and analysethose those datas.
There also is a raw dataset get_around-pricing_project.csv along with a notebook pricingexploration.ipynb that make first tries at creating a regression ML model that would provide a price to user in function of their vehicule features.
There also 4 folder : 
 1. API_app

This folder contains the components required to build and deploy an API on Heroku. API allows for user to get predictionsfrom the Machine Learning wich provided the best results for a price predictions, see API_usage.md for how to use the API.

API link : https://getaround-api-app-82984add3f24.herokuapp.com/

Files detail :

    Dockerfile: Instructions for building a Docker image to deploy the API.
    app.py: Main application file for the API. It handles requests, model inference, and responses.
    requirements.txt: Lists all dependencies required to run the API.
    start.sh: Shell script to start the API server, typically used within Docker.
    test.py: Contains test cases for validating the API's endpoints and functionality.
    __pycache__/: Directory created by Python to store compiled bytecode files for performance optimization. This can be ignored for deployment.

2. catboost_info

This folder holds metadata and logging information for CatBoost model training. It includes details about the model's performance, parameters, and logs generated during training. This folder is useful for debugging and analyzing model behavior.

3. mlflow-heroku-deployment

This directory is set up for deploying the model and tracking experiments using MLflow, a tool for managing machine learning experiments. It is configured for Heroku deployment.

MLFlow app link : https://getaround-app-cb43022eb5de.herokuapp.com/#/experiments/0?searchFilter=&orderByKey=attributes.start_time&orderByAsc=false&startTime=ALL&lifecycleFilter=Active&modelVersionFilter=All+Runs&datasetsFilter=W10%3D

Files detail :

    catboost_info/: Holds information specific to the CatBoost model, similar to the main catboost_info folder.
    mlruns/0: Stores MLflow experiment data, including model artifacts, parameters, metrics, and logs.
    train: Main script or folder used to train models in MLflow experiments.
    Dockerfile: Defines the environment for deploying MLflow on Heroku.
    MLproject: Configuration file for MLflow, defining project structure and dependencies.
    Procfile: Specifies the command to run the application in a Heroku deployment.
    conda.yaml: Specifies the Conda environment configuration used for training or deploying the model.
    requirements.txt: List of dependencies for running MLflow and model training.
    start.sh: Shell script to start the MLflow server or initiate training.

4. streamlit_app

This folder contains the components required for the creation of a Streamlit dashboard that summurize conclusion mase from data analysis on get_around_delay_analysis.xlsx.

Dashboard link : https://getaround-streamlit-dashboard-b538cd0ea51e.herokuapp.com/

Files detail :

    app/: Contains the main Streamlit application code and modules.
    Dockerfile: Instructions to create a Docker image for deploying the Streamlit app.
    get_around_delay_analysis.xlsx: Excel file containing data for analyzing delays in GetAround's service.
    requirements.txt: List of dependencies for the Streamlit application.
    start.sh: Shell script to launch the Streamlit application.
