# Uber/Lyft Cab Price Prediction

A machine learning project that predicts Uber and Lyft ride prices by combining cab trip data with weather conditions. The notebook builds a full preprocessing and training pipeline, then compares three approaches: a baseline Linear Regression model, an XGBoost Random Forest regressor, and a PyTorch neural network.


## Overview

This project uses two datasets:

- `cab_rides.csv` for trip-level cab ride information.
- `weather.csv` for time-based weather observations.

The goal is to predict the ride `price` using trip metadata, time information, and matched weather features.


## Quickstart:

1. Create a directory `data`

2. Download `cab_rides.csv` and `weather.csv` from [Kaggle](https://www.kaggle.com/datasets/ravi72munde/uber-lyft-cab-prices/data?select=cab_rides.csv) into the same directory 

3. Run these following commands:

```bash
chmod +x run_project.sh
./run_project.sh
```

These commands includes creating virtual environment, installing dependancies, and running code for the entire project.


## Project Structure

```bash
├── data/
│   ├── cab_rides.csv
│   └── weather.csv
├── preprocess_data.py
├── nn_model.py
├── nn_training.py
├── prediction.py
├── prediction.ipynb
├── train_test_split.npz
├── nn_checkpoint.pth
├── requirements.txt
└── README.md
```



## File Description

- `data/cab_rides.csv` — Raw ride-share trip data used for model development. 
- `data/weather.csv` — Weather data merged with trip records during preprocessing. 
- `preprocess_data.py` — Script for cleaning, merging, encoding, scaling, and preparing the dataset. 
- `nn_model.py` — Definition of the neural network architecture used for cab price prediction. 
- `nn_training.py` — Training script for fitting the neural network and saving the model checkpoint. 
- `prediction.py` — Script for loading the trained model and generating predictions. 
- `prediction.ipynb` — Notebook for interactive inference, testing, and experimentation. 
- `train_test_split.npz` — Saved train/test arrays for reproducible experiments. 
- `cab_price_checkpoint.pth` — Trained PyTorch model checkpoint. 
- `requirements.txt` — Python dependencies required to run the project. 
- `README.md` — Project documentation and usage instructions. 



## How to Run

1. Put the datasets inside a `data/` directory:
   - `data/cab_rides.csv`
   - `data/weather.csv`

2. Open the notebook:

```bash
jupyter notebook prediction.ipynb
```

3. Run the cells in order.

4. If a compatible GPU is available, XGBoost and PyTorch can use CUDA acceleration.

## Results

The model comparison shows:

| Model | RMSE | MAE | R² |
|------|------|------|------|------|
| Linear Regression | 2.5033 ± 0.0344 | 1.7572 ± 0.0122 | 0.9273 ± 0.0016 |
| XGBoost Random Forest | 1.8834 ± 0.0323 | 1.2576 ± 0.0096 | 0.9589 ± 0.0014 |
| PyTorch Neural Network | 1.7131 ± 0.0323 | 1.1287 ± 0.0087 | 0.9660 ± 0.0013 |

The neural network performs best on the test set, with the lowest error and highest R² among the three models.

## Workflow

### 1. Load the data

The notebook loads both datasets with `pandas`:

- `data/cab_rides.csv`
- `data/weather.csv`

### 2. Impute missing weather data

The `rain` column in the weather dataset contains missing values. These are filled using a KNN-based regression/imputation process.

Steps used:

- Copy the weather dataframe.
- Use numeric predictors:
  - `temp`
  - `clouds`
  - `pressure`
  - `humidity`
  - `wind`
- One-hot encode the `location` column.
- Train a `KNeighborsRegressor` with:
  - `n_neighbors=5`
  - `weights='distance'`
- Scale the features using `StandardScaler`.
- Predict missing `rain` values and write them back into the weather dataframe.

### 3. Merge ride and weather data

The ride and weather datasets are merged using timestamp proximity:

- Ride timestamps are converted from milliseconds.
- Weather timestamps are converted from seconds.
- Both dataframes are sorted by `time_stamp`.
- A nearest-time merge is performed with `pd.merge_asof(...)`.
- Merge tolerance is set to 15 minutes.
- Rows without a weather match are dropped.

### 4. Feature engineering

Additional features are created from the merged dataset:

#### Time-based features
- `hour`
- `day`
- `weekday`

#### Encoded categorical features
The following columns are one-hot encoded:

- `cab_type`
- `source`
- `destination`
- `name`

#### Dropped columns
The following columns are removed before modeling:

- `time_stamp`
- `id`
- `product_id`

### 5. Prepare training data

- Target variable: `price`
- Feature matrix: all remaining columns except `price`
- Features are scaled with `StandardScaler`
- Train/test split:
  - `test_size=0.2`
  - `random_state=42`

## Models

### Linear Regression

A simple baseline regression model is trained using scikit-learn.

#### Metrics
- RMSE: 2.5033 ± 0.0344
- R2: 0.9273 ± 0.0016
- MAE: 1.7572 ± 0.0122

### XGBoost Random Forest

The second model uses `XGBRFRegressor` from XGBoost.

#### Configuration
- `n_estimators=100`
- `max_depth=10`
- `random_state=42`
- `device="cuda"`
- `tree_method="hist"`

#### Metrics
- RMSE: 1.8834 ± 0.0323
- R2: 0.9589 ± 0.0014
- MAE: 1.2576 ± 0.0096

### PyTorch Neural Network

The notebook also defines and trains a deep neural network for price prediction.

#### Architecture

```python
Input
→ Linear(input_dim, 1024)
→ ReLU
→ BatchNorm1d(1024)
→ Linear(1024, 512)
→ ReLU
→ Dropout(0.2)
→ Linear(512, 32)
→ ReLU
→ Linear(32, 1)
```

#### Training setup
- Framework: PyTorch
- Loss function: `nn.MSELoss()`
- Optimizer: Adam
- Learning rate: `0.001`
- Epochs: `800`
- Device: CUDA if available, otherwise CPU

#### Checkpointing
The neural network training saves checkpoints to:

```bash
nn_checkpoint.pth
```

If the checkpoint file already exists, training resumes from the saved state.

#### Neural network metrics
- RMSE: 1.7131 ± 0.0323
- R2: 0.9660 ± 0.0013
- MAE: 1.1287 ± 0.0087


## Visualization

The notebook includes residual scatter plots for model predictions:

- Prediction error vs. data point index
- Horizontal reference line at zero
- Used to inspect model error distribution visually


## Notes

- Weather data is aligned to ride data using nearest timestamps rather than exact matches.
- Missing weather values for `rain` are handled before merging.
- The notebook uses both traditional machine learning and deep learning methods for comparison.
- GPU support is enabled where available.

## Future Improvements

Possible next steps for this project:

- Add more feature engineering such as month, weekend, or holiday indicators.
- Tune hyperparameters for XGBoost and the neural network.
- Use cross-validation for more robust model evaluation.
- Save the preprocessing pipeline and trained models for deployment.
- Build an inference script or API for real-time price prediction.

## License

This project is for educational and research purposes.