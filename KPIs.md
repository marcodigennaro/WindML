# KPI lists

- **This package**
- A list of **jupyter notebooks**
  1. Scalability
  2. Read_One
  3. Read_All
- **Scalability**
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


- **Read_One**
  1. Read and Clean data, check for missing values and outliers 
    1.1. Check for emtpy features: There is no empty column, so we can keep all features.
    1.2 Check uniformity of Database with respect to month, day and hour: The distribution of data is uniform.
    1.3 Exploratory Data Analysis (EDA)
    1.4 Find univariate outliers using box plots
      - Pitch_angle (Ba_avg):
        - Discretization issue with 'Ba_avg' -> introducing 'rounded_Ba_avg'
        - Removing outliers (keeping only -5 < Ba < 5)



- **Prediction Accuracy**: 

| Metric     | Linear | Polynomial | Kernel |
|------------|--------|------------|--------|
| MAE        | xx     | 2.xx       | 0.xx   |
| RMSE       | xx     | 2.xx       | 1.xx   |
| R-squared  | xx     | 0.xx       | 0.xx   |
| R-squared  | xx     | 0.xx       | 0.xx   |


- **Model Training Time**: 
- Track the time taken to train the machine learning models on the enriched dataset. 
- This indicates the efficiency of the model training process and scalability of the solution.


| Metric     | Linear | Polynomial | Kernel |
|------------|--------|------------|--------|
| Model Training Time        | xx     | 2.xx       | 0.xx   |

- **Feature Importance**: Determine the importance of different features, including additional parameters like snowfall_1h, rainfall_1h, air density, and humidity, in predicting turbine output. Understanding feature importance provides insights into the factors influencing turbine performance.
- **Generalization**: Evaluate the generalization ability of the models by testing them on unseen data from different turbines or locations. Split the dataset into training and testing sets and measure model performance on the test set.
- **Value Proposition**: Quantify the potential value that the machine learning models can bring to Engie in terms of cost savings and maintenance efficiency. Consider scenarios where unexpected behavior is detected early, leading to proactive maintenance actions.
- **Engagement with Client**: Measure the level of engagement and interest from Engie's Head of Wind Turbines during the meeting. Assess their feedback, questions, and willingness to explore further collaboration opportunities based on the presented results and value proposition.
- **Interpretability**: Evaluate the interpretability of the machine learning models, especially in a domain like wind turbine operations where explainability is crucial for decision-making. Ensure that the models provide insights into the factors influencing turbine performance.
- **Model Robustness**: Assess the robustness of the machine learning models against variations in input data, such as changes in weather conditions or sensor noise. Evaluate model performance under different scenarios to ensure reliability in real-world applications.
- **Operational Efficiency**: Measure the efficiency gains achieved through the use of machine learning models for performance monitoring and maintenance scheduling. Quantify reductions in downtime and maintenance costs attributable to the predictive capabilities of the models.
