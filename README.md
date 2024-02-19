## Structure

|-- ARIMA  
  - |--  weightselection  
  - |--  test result  
  - |--  submitted result

|-- code  
  - |-- code  
    - |-- select weights

|-- GRU  
  - |-- weightselection  
  - |-- rate model  
  - |-- submitted result  
    - |-- submitted model  
    - |-- rate result  

|-- LSTM  
  - |-- result for regions  
  - |-- weightselection  
      -|-- region  
  - |-- rate model  
  - |-- submitted result  
  - |-- submitted model  
  - |-- model for regions  
  - |-- rate result

## Usage & Explanation

This repository is organized to facilitate easy access to both the code used for model development and the outcomes of these models, including weight selections, test results, and submitted results for various forecasting models like ARIMA, GRU, and LSTM. Here's a brief guide on how to navigate and use the repository:

### Code Directory

- **`/code/code/`**: Contains all scripts and notebooks for model development, including data preprocessing, training, evaluation, and weight selection processes. The `select weights` subdirectory is specifically dedicated to scripts that facilitate the selection of model weights, but we can ignore that part since it's not well developed.

### Model and Results Directories

For each model type (ARIMA, GRU, and LSTM), there are dedicated directories structured to neatly separate the components and stages of the modeling process:

- **Model Directories (ARIMA, GRU, LSTM)**: Include subdirectories for the selection of weights (`weightselection`) (ignore that part), storage of test models (`rate model`), and final results prepared for submission (`submitted result`). The `submitted model` subdirectories house the finalized models, whereas the `rate result` directories contain the output from test models.

- **LSTM Specifics**: 
  - `result for regions` and `model for regions` cater to region-specific forecasting, offering localized predictions.
  - `weightselection/region` provides insights and files related to the weight selection for these region-focused models (ignore).

### Data Storage

- **`/code`**: Serves as the central hub for all training data used across various models, ensuring easy access and management of datasets critical for model training.
- In the root directory of this repository, you will find the final results that were submitted to the CDC.

## Notebook Descriptions

### STATE-LEVEL MODEL `exp smoothing for each state.ipynb`

This Jupyter notebook is dedicated to forecasting state-level flu admissions across the United States, employing two distinct statistical methodologies: ARIMA (AutoRegressive Integrated Moving Average) and Exponential Smoothing. The choice of method is tailored to the data characteristics and availability for each state, ensuring the most reliable forecasting results under varying conditions.

#### Overview:

- **ARIMA Forecasts**: The notebook utilizes ARIMA models for predicting the next four weeks of flu admissions in nearly all states. This approach benefits from incorporating multiple features into the model, including historical flu admission rates alongside other relevant predictors, to enhance forecast accuracy.

- **Exponential Smoothing for DC**: Due to data limitations specific to the District of Columbia (DC), the notebook adopts Exponential Smoothing for this jurisdiction. This method focuses solely on the admission rate, applying a time series forecasting technique that accounts for trends and seasonality in the data without the need for additional predictors.

#### Key Features:

- **State-Specific Forecasts**: Tailored forecasting models for each state, with specific adjustments for DC, to accommodate unique data challenges and improve local prediction accuracy.
- **Methodological Comparison**: Insightful comparison between ARIMA and Exponential Smoothing methods, demonstrating their application and performance across different data environments.
- **Four-Week Horizon**: Forecasts extend over a four-week period, providing short-term insights crucial for public health planning and response efforts.

#### Usage:

This notebook is intended for epidemiologists, public health officials, and researchers interested in flu trend forecasting at the state level. 

## REGION-LEVEL MODEL `BaselineModelsforregions.ipynb`

This Jupyter notebook focuses on employing Long Short-Term Memory (LSTM) networks for forecasting state-level flu admissions across the United States. The innovative approach involves training a unique LSTM model for each Health and Human Services (HHS) region, leveraging these models to predict the flu admission rates for the states within each region over the next four weeks.

### Overview

The notebook is structured around the development and application of LSTM models tailored to the specificities of each HHS region. It aims to capture regional trends and patterns in flu admissions, thereby enhancing the accuracy of state-level forecasts within those regions.

#### Region Mapping:
The United States is divided into several HHS regions, each encompassing a group of states identified by their FIPS codes. The LSTM models are trained on data aggregated at these regional levels:

- **Region 1 - Boston:** Includes states with FIPS codes 09, 23, 25, 33, 44, 50.
- **Region 2 - New York:** Comprises states 34 and 36.
- **Region 3 - Philadelphia:** Encompasses states 10, 24, 42, 51, 54.
- **Region 4 - Atlanta:** Contains states 01, 12, 13, 21, 28, 37, 45, 47.
- **Region 5 - Chicago:** Includes states 17, 18, 26, 27, 39, 55.
- **Region 6 - Dallas:** Covers states 05, 22, 35, 40, 48.
- **Region 7 - Kansas City:** Comprises states 19, 20, 29, 31.
- **Region 8 - Denver:** Encompasses states 08, 30, 38, 46, 49, 56.
- **Region 9 - San Francisco:** Contains states 04, 06, 15, 32.
- **Region 10 - Seattle:** Includes states 02, 16, 41, 53.

#### Important Consideration:
It's critical to note that the current regional classifications may not fully capture the underlying trends in flu admissions. The regions, as defined, serve as an initial framework for model training and will be subject to reevaluation and potential reclassification to better align with observed admission patterns.

#### How It Works (Training Part)

1. **Data Preparation**: 
    - The data for each state is extracted and formatted into sequences (`seq_length`) that serve as input to the LSTM model, alongside corresponding labels (`output_size`) representing the future flu admission rates we aim to predict.
    - The dataset is filtered based on `region_fips` to ensure that each LSTM model is trained on data from states belonging to a specific HHS region. This regional focus allows the models to capture unique geographical and temporal patterns in flu trends.

2. **Model Training**:
    - A custom LSTM class is defined to handle sequences of flu admission data, with the architecture configured to accommodate various `input_size`, `hidden_layer_size`, and `num_layers` parameters for experimentation.
    - For each region, the LSTM model is trained using historical flu admission data, applying MinMax scaling for normalization to improve model performance.
    - The training process involves splitting the data into training and validation sets, iterating over multiple epochs, and employing early stopping based on validation loss to prevent overfitting.

3. **Prediction and Evaluation**:
    - Post-training, the LSTM models are employed to forecast flu admission rates for the forthcoming four weeks across each state within their respective regions.
    - Model predictions are inverse-transformed to revert the normalization process, enabling direct comparison with actual flu admission rates.
    - The performance of each regional model is evaluated based on mean squared error (MSE) metrics for both overall and weekly predictions, facilitating an understanding of model accuracy and reliability.

4. **Summary and Output**:
    - Results, including MSE metrics and model configurations, are compiled into a summary DataFrame, which is then exported to a CSV file for further analysis and review.
    - Predictions for each region are also saved in CSV format, providing accessible forecasts that can inform public health decisions and strategies.

#### Prediction Process **Run Best Model for 100 Times**

Using the `LSTM` class defined in our repository, we train models tailored to the unique characteristics of flu trends within each of the ten designated HHS regions. After identifying the best model configuration for each region based on validation loss, we proceed with the prediction phase. Here's how it works:

1. **Data Preparation**: We start by loading the training data and applying MinMax scaling to normalize the features, ensuring that our LSTM models can learn more effectively from the data.

2. **Model Selection and Loading**: For each region, we load the LSTM model that demonstrated the best performance during the training phase. These models are stored in the `/LSTM/model for regions/` directory and are identified by their specific configuration parameters (e.g., sequence length, number of layers).

3. **Running Predictions**: Each selected model is run 100 times on the test dataset to generate forecasts for the next four weeks. This repeated prediction process helps account for the stochastic nature of the LSTM's predictions, providing a more comprehensive view of potential future trends.

4. **Aggregating and Saving Results**: The predictions from these 100 runs are aggregated and saved to the `/LSTM/result for regions/` directory. This dataset offers a rich source of information for analyzing the forecasted flu admission rates and can be used to calculate confidence intervals or perform other statistical analyses.

#### Key Components

- **`LSTM` Class**: Defines the LSTM model architecture, including dropout layers to mitigate overfitting.
- **`prepare_data_main_model` Function**: Processes the raw flu admission data into a format suitable for LSTM training, considering specified sequence lengths and output sizes.
- **`splitdata` Function**: Segregates the prepared data into training and testing datasets based on a provided ratio, supporting model evaluation.
- **`RunBuilder` Class**: Facilitates the generation of model configurations, enabling systematic experimentation with different hyperparameters.

This approach underscores the potential of LSTM networks in capturing complex temporal dependencies in flu admission data, presenting a scalable framework for regionalized flu forecasting.


## US-LEVEL MODEL `BaselineModelsforallstate.ipynb`

### Overview

The `BaselineModelsforallstate.ipynb` notebook extends our forecasting efforts to a national scale, employing both Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks to predict state-level flu admissions across the entire United States. This approach mirrors the methodology applied in our regional models, adapting it to capture trends and variations in flu admissions at a countrywide level.

### Approach

- **LSTM and GRU Models**: The notebook explores the use of two powerful recurrent neural network architectures, LSTM and GRU, known for their effectiveness in handling time series data. By comparing these models, we aim to identify the most accurate forecasting method for national flu trends.
- **Data Preparation and Normalization**: Similar to the regional models, the data preparation phase involves scaling the features to a normalized range to facilitate model training. The sequence length, output size, and historical data range are carefully selected to optimize model performance.
- **Training Process**: Each model is trained on historical flu admission data, with parameters fine-tuned to best capture the underlying patterns at a national level. The training process leverages a split of the data into training and validation sets, employing early stopping to prevent overfitting.
- **Prediction and Evaluation**: Post-training, the models are utilized to forecast flu admission rates for the forthcoming four weeks. Predictions are generated in multiple runs to assess the models' consistency and reliability.

### Key Components

- **Model Architecture**: The notebook details the construction of LSTM and GRU models, including layer configurations, dropout rates for regularization, and the optimization algorithm.
- **Model Training and Validation**: It includes a comprehensive guide to training the models, selecting hyperparameters, and evaluating their performance on validation data.
- **Forecast Generation**: The methodology for using trained models to generate and save forecasts is outlined, ensuring that predictions are readily available for analysis and decision-making.

### Utilization

This US-level forecasting model serves as a crucial tool for public health officials, researchers, and policymakers. By providing accurate and timely predictions of flu admissions, it supports strategic planning and resource allocation to mitigate the impact of flu seasons across the United States.

For further details on the model architecture, training process, and prediction methodology, please refer to the `BaselineModelsforallstate.ipynb` notebook within the repository.



## Ensemble Results `checkprediction.ipynb`

### Overview

The `checkprediction.ipynb` notebook is central to synthesizing our predictive modeling efforts, taking the output from deep learning models and refining it through several key steps to produce and visualize the final forecasts. This notebook performs an ensemble of different model outputs, converts rate predictions to actual admission numbers, adapts deep learning model results to fit a quantile forecasting framework, and selects optimal weights to aggregate forecasts into a consensus prediction.

### Key Processes

- **Conversion of Rate to Admissions**: Initially, the notebook takes the predicted rates of flu admissions and converts them into actual admission numbers, providing a more tangible metric for health officials and policymakers.

- **Quantile Model Conversion**: It then transforms the deep learning model outputs into quantile forecasts, enabling the estimation of uncertainty in the predictions and offering a comprehensive view of potential future outcomes.

- **Weight Selection and Ensemble**: Through a selection of weights for each model's output based on their historical performance, the notebook combines forecasts from four different models into a single ensemble result. This process leverages the strengths of each model to improve overall forecast accuracy and reliability.

- **Final Result Generation and Visualization**: Finally, the notebook generates the consolidated forecast results and visualizes them in various plots. These visualizations help in interpreting the models' predictions, comparing them against actual data, and understanding the forecasted trends and their potential implications.

### Utilization

This notebook is designed for data scientists, epidemiologists, and public health officials who are looking for a refined approach to flu forecasting. By incorporating ensemble methods and quantile predictions, it offers a robust framework for anticipating flu admissions and preparing more effectively for future flu seasons.

### Features

- **Visualization**: It includes plotting functions to visually compare the ensemble predictions against historical data, providing an intuitive understanding of the forecast accuracy and potential flu trends.
- **Flexibility**: While the notebook is tailored for flu admission forecasting, the methodologies it employs can be adapted to other diseases or conditions where predictive modeling is applicable.

For detailed instructions on running the `checkprediction.ipynb` notebook, as well as explanations of the code and methodologies used, please refer to the notebook within the repository.

