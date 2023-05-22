import pandas as pd
import numpy as np
import os

from datetime import datetime


class TimeSeries:
    """
    A class used to represent a TimeSeries for forecasting.

    ...

    Attributes
    ----------
    name : str
        A name for easy identification of the results (like the type of base stock)
    series : np.array
        The numerical data of the series
    date_index : pd.DatetimeIndex
        a pandas DatetimeIndex representing the index of the original pandas series
    train_window_size : int
        size of the window to be used for training the model
    predict_window_size : int
        size of the window to be used for making predictions
    window : int
        total size of the window (train_window_size + predict_window_size)
    n_predictions : int
        the number of predictions that can be made given the series and window sizes
    train_windows : np.array
        2D numpy array where each row is a training window from the series
    predict_windows : np.array
        2D numpy array where each row is a prediction window from the series
    predictions : np.array
        2D numpy array to store the model's predictions
    residuals : np.array
        1D numpy array of residuals/errors where each value represents a window
    abs_residuals : np.array
        the absolute values of the residuals/errors
    abs_mean_error : float
        float representing the mean absolute error of the model's predictions over all the windows

    Methods
    -------
    create_windows():
        Creates training and prediction windows from the series using the values provided at object instantiation.
    evaluate_model():
        Evaluates the model's performance by calculating the mean absolute error.
    save_data(model, filepath='data.csv'):
        Saves the model's metadata and performance metrics to a CSV file.
    """

    def __init__(self, name, series, train_window_size, predict_window_size):
        """
        Parameters
        ----------
            name : str
                A name for easy identification of the results (like the type of base stock)
            series : pd.Series
                The pandas series object with a datetime index
            train_window_size : int
                The size of the window for training the model
            predict_window_size : int
                The size of the window for making predictions
        """

        self.name = name
        self.series = series.values
        self.date_index = series.index
        self.train_window_size = train_window_size
        self.predict_window_size = predict_window_size
        self.window = train_window_size + predict_window_size
        self.n_predictions = len(series) - self.window + 1

    def create_windows(self):
        """
        Creates training and prediction windows from the series.

        Returns
        -------
        response : dict
            A dictionary containing training and prediction windows.
        """
        # Initialize the train and predict windows
        train_windows = np.zeros((self.n_predictions, self.train_window_size))
        predict_windows = np.zeros((self.n_predictions, self.predict_window_size))
        predictions = predict_windows.copy() # Predictions are the same size as actuals

        # Iterate over the series to populate the windows
        for i in range(self.n_predictions):
            train_windows[i] = self.series[i:i+self.train_window_size]
            predict_windows[i] = self.series[i+self.train_window_size:i +
                                             self.train_window_size+self.predict_window_size]

        # Save windows as class attributes for easy access
        self.train_windows = train_windows
        self.predict_windows = predict_windows
        self.predictions = predictions

        # preparing a response. Because we're not animals
        response = dict(train_windows=train_windows, predict_windows=predict_windows)
        return response
    
    # Calculate the error, also called the residual
    def mean_absolute_error(self):
        """
        MAE - Provides the mean absolute error for the model's predictions.
        """

        abs_residuals = np.abs(self.residuals)
        mae = abs_residuals.mean()

        return mae
    
    def mean_squared_error(self):
        """
        MSE - Provides the mean absolute error for the model's predictions.
        """
        squared_residuals = self.residuals**2
        mse = squared_residuals.mean()
        
        return mse
    
    def root_mean_squared_error(self):
        """
        RMSE - Provides the root mean squared error for the model's predictions.
        """
        rmse = np.sqrt(self.mean_squared_error())
        return rmse

    def error_1st_prediction(self):
        """
        MAE_1st, MSE_1st - Provides the mean absolute error and mean squared error
        for the model's 1st prediction.
        """
        residuals_1st = self.residuals[:, 0]
        abs_residuals_1st = np.abs(residuals_1st)
        mae_1st = abs_residuals_1st.mean()
        mse_1st = np.mean(residuals_1st**2)
        
        return (mae_1st,mse_1st)

    def error_2nd_prediction(self):
        """
        MAE_2nd, MSE_2nd - Provides the mean absolute error and mean squared error
        for the model's 2nd prediction.'
        """
        residuals_2nd = self.residuals[:, 1]
        abs_residuals_2nd = np.abs(residuals_2nd)
        mae_2nd = abs_residuals_2nd.mean()
        mse_2nd = np.mean(residuals_2nd**2)
        
        return (mae_2nd,mse_2nd)
    
    def error_3rd_prediction(self):
        """
        MAE_3rd, MSE_3rd - Provides the mean absolute error and mean squared error
        for the model's 3rd prediction.
        """
        residuals_3rd = self.residuals[:, 2]
        abs_residuals_3rd = np.abs(residuals_3rd)
        mae_3rd = abs_residuals_3rd.mean()
        mse_3rd = np.mean(residuals_3rd**2)
        
        return (mae_3rd,mse_3rd)

    def mean_absolute_percentage_error(self):
        """
        MAPE - Provides the mean absolute percentage error for the model's predictions.
        Note: Be cautious of division by zero when using this metric.
        """
        mape = np.mean(np.abs((self.residuals / self.predict_windows)) * 100)
        return mape

    def mean_absolute_scaled_error(self):
        """
        MASE - Provides the mean absolute scaled error for the model's predictions.
        Assumes the naive forecasting method of the previous observation.
        """
        naive_forecast_residuals = self.series[self.train_window_size:-1] - self.series[self.train_window_size-1:-2]
        mae_naive = np.mean(np.abs(naive_forecast_residuals))
        mase = self.mae / mae_naive
        return mase

    def symmetric_mean_absolute_percentage_error(self):
        """
        sMAPE - Provides the symmetric mean absolute percentage error for the model's predictions.
        This version of MAPE handles zeros in the actual values better than MAPE.
        """
        numerator = np.abs(self.predict_windows - self.predictions)
        denominator = (np.abs(self.predict_windows) + np.abs(self.predictions)) / 2
        smape = np.mean(numerator / denominator) * 100
        return smape

    def mean_directional_accuracy(self):
        """
        MDA - Provides the mean directional accuracy for the model's predictions.
        This measure shows the proportion of forecasts that correctly predict the direction of change.
        """
        actual_direction = np.sign(self.predict_windows[1:] - self.predict_windows[:-1])
        forecast_direction = np.sign(self.predictions[1:] - self.predictions[:-1])
        mda = np.mean(actual_direction == forecast_direction)
        return mda

    def evaluate_model(self):
        """
        Evaluates the model's performance by calculating several metrics.

        """
        actuals = self.predict_windows
        forecast = self.predictions
        self.residuals = actuals - forecast
        
        self.mae = round(self.mean_absolute_error(),3)
        self.mse = round(self.mean_squared_error(),3)
        self.rmse = round(self.root_mean_squared_error(),3)
        self.mae_1st = round(self.error_1st_prediction()[0],3)
        self.mse_1st = round(self.error_1st_prediction()[1],3)
        self.mae_2nd = round(self.error_2nd_prediction()[0],3)
        self.mse_2nd = round(self.error_2nd_prediction()[1],3)
        self.mae_3rd = round(self.error_3rd_prediction()[0],3)
        self.mse_3rd = round(self.error_3rd_prediction()[1],3)
        self.mape = round(self.mean_absolute_percentage_error(),3)
        self.mase = round(self.mean_absolute_scaled_error(),3)
        self.smape = round(self.symmetric_mean_absolute_percentage_error(),3)
        self.mda = round(self.mean_directional_accuracy(),3)

        return {"MAE": self.mae, "MSE": self.mse, "RMSE": self.rmse,
                "MAE_1st": self.mae_1st, "MAE_2nd": self.mae_2nd, "MAE_3rd": self.mae_3rd,
                "MSE_1st": self.mse_1st, "MSE_2nd": self.mse_2nd, "MSE_3rd": self.mse_3rd,
                "MAPE": self.mape, "MASE": self.mase, "sMAPE": self.smape, "MDA": self.mda}
 
    def predict(self, predict_func):
        """
        Uses the given predict_func to make predictions.

        Parameters
        ----------
            predict_func : function
                A function that takes the train_windows from the TimeSeries instance and returns predictions.
        """
        self.predictions = predict_func(self.train_windows)       
        
        
    def save_data(self, model, filepath='data.csv'):
        """
        Saves the model's metadata and performance metrics to a CSV file.

        Parameters
        ----------
        model : str
            The name or type of the model used.
        filepath : str, optional
            The file path where the data will be saved (default is 'data.csv')

        Returns
        -------
        df : pd.DataFrame
            A pandas DataFrame containing the saved data.
        """

        # Save the data to a csv file
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        

        data = dict(
            time_series=[self.name],
            model=[model],
            train_window_size=[self.train_window_size],
            predict_window_size=[self.predict_window_size],
            mae=[self.mae],
            mse=[self.mse],
            rmse=[self.rmse],
            mae_1st=[self.mae_1st],
            mse_1st=[self.mse_1st],
            mae_2nd=[self.mae_2nd],
            mse_2nd=[self.mse_2nd],
            mae_3rd=[self.mae_3rd],
            mse_3rd=[self.mse_3rd],
            mape=[self.mape],
            mase=[self.mase],
            smape=[self.smape],
            mda=[self.mda],
            timestamp=[now]
        )

        df = pd.DataFrame(data=data)
        # Appends data to output file if it exists. Otherwise, creates it.

        if os.path.isfile(filepath):
            df.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df.to_csv(filepath, index=False, header=True)

        return df
    


# Define a function to use as our predict_func
def baseline(train_windows, predict_window_size):
    """
    A baseline prediction function. 

    Predicts the last value from the training window for each step in the prediction window.

    Parameters
    ----------
        train_windows : np.array
            2D numpy array where each row is a training window from the series

    Returns
    -------
        predictions : np.array
            2D numpy array where each row is the predicted values for each prediction window
    """

    last_values = train_windows[:, -1]
    return np.transpose([last_values]*predict_window_size)


def create_dataset():
    # Create date range from Jan 2007 to Dec 2022
    dates = pd.date_range(start='2007-01-01', end='2022-12-01', freq='M')
    
    # Create synthetic oil price data
    np.random.seed(42)  # Seed for reproducibility
    
    # Define sine wave parameters to simulate cyclical changes
    amp = 20  # Amplitude
    period = 6*12  # Period in months
    
    # Define trend parameters
    slope = 0.15
    intercept = 50
    
    # Define noise parameters
    mu = 0
    sigma = 5
    
    # Generate synthetic oil prices
    prices = amp * np.sin(2 * np.pi * (np.arange(len(dates)) / period)) + slope * np.arange(len(dates)) + intercept + np.random.normal(mu, sigma, len(dates))
    
    # Convert to pandas series
    return pd.Series(prices, index=dates, name='Oil Prices')
