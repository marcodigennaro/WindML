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

Training on time data only does not give good results:
![plot](https://github.com/marcodigennaro/WindML/blob/main/images/XGBR.jpeg?raw=true)

## **Machine Learning**

- **Prediction Accuracy**: 

| Metric    | Linear             | Polynomial         | Kernel             |
|-----------|--------------------|--------------------|--------------------|
| MAE       | 123.83655569129044 | 26.563523071212337 | 18.069286899183126 |
| RMSE      | 28737.204834443808 | 2105.0001314111532 | 1699.991079011466  |
| R-squared | 0.8546457205691437 | 0.9895920141194349 | 0.9912831790974509 |


- **Model Training Time**: 
- Track the time taken to train the machine learning models on the enriched dataset. 
- This indicates the efficiency of the model training process and scalability of the solution.


| Metric              | Linear | Polynomial | Kernel  |
|---------------------|--------|------------|---------|
| Model Training Time | 752 ms | 939 ms     | 5min 2s |


