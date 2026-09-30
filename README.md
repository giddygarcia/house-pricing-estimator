# 🏡 House Pricing Predictive Modelling

## Overview

**Tech Stack:** **Python, Docker, FastAPI, React, Google Cloud Platform (GCP)**, Pandas, NumPy, Scikit-learn, LightGBM, XGBoost, Statsmodels, Matplotlib, Seaborn, uv

This repository focuses on **building and deploying a predictive machine learning model on Portugal house pricing data**. The [Real Estate Listings in Portugal dataset](https://www.kaggle.com/datasets/luvathoms/portugal-real-estate-2024) covers 6 years (2018 to 2024) of data collected from real listings on several Portugese real estate websites. 

**The model was successfully deployed on GCP** as a containerised price prediction model that served live predictions through a web application.

*Note: The GCP deployment is no longer live due to hosting costs and constraints. The model and deployment code remain fully available in this repository for recreation.*

<div align="center">
  <img src="visuals/demo-image.png" alt="Model demo" width="50%"/>
</div>


## Objectives:
1. Model Training and Evaluation: Train a regression model on the Portugal Housing prices
    * Evaluate different regression tree and boosted models
    * Practice ensemble modelling on the data
    * Determine the best type of model
2. Feature Importance: List which features are the most important to pricing
3. Full ML Pipeline to Deployment: Deploy the model using Docker and GCP
  

<div align="center" style="width: 95%; margin: auto;">

<video src="visuals/demo-vid.mp4" alt="Video demo" controls width="100%"></video>

</div>

## Key Findings:
* Cleaned and consolidated unstructured, disorganized, multi-source data into one unified, ML-ready dataset. **Resolved issues that affected 39% of records.**
* From the baseline, **boosted models gave up to ~€128k euro deduction in MAE** / Mean Absolute Error for a vast improvement in predictive performance.
* **Light Gradient Boosting** was chosen for deployment. It effectively gave the best performance overall and is practical to deploy as a strong singular model. 
* ***Stacked generalization did not show marginal improvements***: Light Gradient Boosting outperformed it and XGB also competed closely with the meta learner. 
* The 5 most important features for pricing are:
    1. NumberOfBathrooms - Essentially our proxy variable for overall number of rooms, property size and functionality - all these details it can suggest makes it the strongest driver for price levels in our deployed LGB model
    2. Town - the smallest most specific geographical scale is the 2nd important driver for price predictions.
    3. City - geographical variables take hold of 2 feature importance spots. Overall location of a house in Portugal is extremely important for pricing predictions. 
    4. LivingArea - measurable indoor livable space is 4th important.
    5. EnergyCertNum - the energy certificate of a house and its efficiency according to Portugal standards lend strong prediction for price points

## 📦 Libraries used:
* pandas
* numpy
* matplotlib
* seaborn
* scikitlearn
* statsmodels

### 🔎 Viewing / Installation:
* *Viewing Option:* For complete analysis and demonstration, simply view the notebook file `RealEstateAnalysis.ipynb`.
* For recreation and development:
    1. Clone this repo & install dependencies: 
        ```bash
        git clone https://github.com/giddygarcia/house-pricing-estimator.git
        cd house-pricing-estimator
        uv sync
        ```
    2. **Model Recreation:** Run `train.py` found inside the "api" folder to rebuild and refit the Light Gradient Boosting model used in this research.  
        ```bash 
        uv run python api/train.py
        ```

## ✉️ Author and Contact Information
Developed by: Christine Garcia 

Have questions? Feel free to:
* email me at cavgarcia22@gmail.com 
* connect on [LinkedIn](www.linkedin.com/in/cavgarcia) 
* or [view more projects](https://github.com/giddygarcia) that I enjoyed making