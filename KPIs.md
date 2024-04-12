# KPI lists

- **This package**
- A list of **jupyter notebooks**
  1. Column names analysis
  2. Scalability
  3. Data Analysis
  4. Time Series and Forecast
  5. Machine Learning 

## **Scalability**
The scalability of the developed models was assessed by evaluating computational resources required as dataset size increases for four different python libraries, namely: Pandas, Dask, Vaex and Modin.

The results are shown below (test results from Processor: 3,1 GHz Dual-Core Intel Core i5 with Memory: 8 GB 2133 MHz)

| File       | Size (MB) |
|------------|-----------|
| R80736.csv | 51.41     |
| R80721.csv | 51.20     |
| R80790.csv | 51.03     |
| R80711.csv | 51.68     |

| Library | Time taken (sec) | Max memory usage (MB) |
|---------|------------------|-----------------------|
| pandas  | 6.13             | 735.23 MB             |
| dask    | 18.38            | 603.20 MB             |
| vaex    | 3.11             | 268.40 MB             |
| modin   | 11.60            | 245.38 MB             |

**Conclusion**:
  - Time Efficiency: Vaex and Pandas show best results,
  - Memory Efficiency: Modin and Vaex show best results,
  - Ease of Use and Integration: Pandas is more available.

## **Data Analysis**
  1. Read and Clean data, check for missing values and outliers 
     1. Check for emtpy features: There is no empty column, so we can keep all features.
     2. Check uniformity of Database with respect to month, day and hour: The distribution of data is uniform.
     3. Exploratory Data Analysis (EDA)
     4. Find univariate outliers using box plots
        - Pitch_angle (Ba_avg):
          - Discretization issue with 'Ba_avg' -> introducing 'rounded_Ba_avg'
          - Removing outliers (keeping only -15 < Ba < 15)

![plot](https://github.com/marcodigennaro/WindML/blob/main/images/windrose.jpeg?raw=true)
![plot](https://github.com/marcodigennaro/WindML/blob/main/images/heatmap.jpeg?raw=true)

## **Time Series Analysis**

### 1. Calculate Produced Energy and Capacity Factor
In this part we calculate three quantities to be analysed as a funciton of the time:

- The **Average Energy** can be calculated from the average Active Power (P_avg, in kW)
- The **Produced Energy** is the integral over time of the average energy
- The **Capacity Factor** is the ratio of the real power output and nominal rated power

### 2. Test Auto-Regression analysis
- Auto-Regression: What if we could just find a stationary explaination for the data series?

### 3. Test Regularizing gradient boosting framework with XGBOOST
- Feature importance analysis reveals that the month is the most important variable

### Conclusions

- Predicting the value of the produced power by using its own value only does not produce relevant results

![plot](https://github.com/marcodigennaro/WindML/blob/main/images/AR.jpeg?raw=true)
![plot](https://github.com/marcodigennaro/WindML/blob/main/images/XGBR.jpeg?raw=true)

## **Machine Learning**


Desired output of the turbine: 
- rotor torque
- converter torque
- power output

Only Active Power (P_avg) and Rotor Torque (Rm_avg) are availeble. 

The list of features used for learning is:
```[ 'Ws_avg', 'Rs_avg', 'Yt_avg', 'Ba_avg']```

### 1. Create a set of pipelines & define a parameter grid to optimize kernels

- Linear
- Polynomial
- Kernel Ridge Regression

### 2. Compare the different models, once optimized, versus the size of the database


- **Prediction Accuracy**: 

| Metric    | Linear             | Polynomial         | Kernel             |
|-----------|--------------------|--------------------|--------------------|
| MAE       | 123.83655569129044 | 26.563523071212337 | 18.069286899183126 |
| RMSE      | 28737.204834443808 | 2105.0001314111532 | 1699.991079011466  |
| R-squared | 0.8546457205691437 | 0.9895920141194349 | 0.9912831790974509 |


- **Model Training Time**: 

| Metric              | Linear | Polynomial | Kernel  |
|---------------------|--------|------------|---------|
| Model Training Time | 752 ms | 939 ms     | 5min 2s |


### 3. Consider the contribution of 'enriched variables'

- Adding ```['rain_1h', 'snow_1h', 'humidity']``` to the descriptor does not improve the system prediction accuracy.

### Conclusion

- Wind speed, Rotor speed, Nacelle temperature and Pitch angle are enough to build a Machine Learning Model
- Kernel models are notably heavy in memory, especially KRR ~O(N^2). This limitation imposed me to limit the training on 20000 entries
![plot](https://github.com/marcodigennaro/WindML/blob/main/images/learning_curve.jpeg?raw=true)
![plot](https://github.com/marcodigennaro/WindML/blob/main/images/learning_curve_enriched.jpeg?raw=true)

