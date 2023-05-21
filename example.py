import utility as ut

oil_prices = ut.create_dataset()

# Print and plot the data
print(oil_prices.head())
oil_prices.plot()

# Instantiate TimeSeries with your data and window sizes
ts = ut.TimeSeries(name='random_oil_prices', series=oil_prices, train_window_size=30, predict_window_size=3)
ts.create_windows()

# Make predictions using the baseline_predict function
ts.predict(lambda x: ut.baseline(x, ts.predict_window_size))

# Evaluate the model
ts.evaluate_model()

ts.save_data('baseline')
