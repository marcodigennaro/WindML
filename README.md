# WindML

This repo provides a tutorial of wind turbine performance prediction based on weather parameters. 

Data are based on a subset of the ENGIE open dataset.

## Getting Started

### Prerequisites

- Python ^3.8
- Poetry 

### Getting started

```
# Navigate to your local folder
cd /your/local/folder

# Clone the WindML repository
git clone https://github.com/marcodigennaro/windml  

# Enter the folder
cd windml/

# Install the package
poetry install

# Activate the environment
source .venv/bin/activate

# Start Jupyter Lab
jupyter-lab  
```

Run any of the jupyter notebooks to visualize data and perform ML algorithms.

### Data

Data are available at this [URL](https://opendata-renewables.engie.com/pages/home/).
Since this is not always functioning, a `data` folder was included in this package.

### Content of the Jupyter Notebooks

  1. Scalability

     - Tests the memory/speed performances of 4 python libraries
          The results are shown below (test results from Processor: 3,1 GHz Dual-Core Intel Core i5 with Memory: 8 GB 2133 MHz)
          
          | File       | Size (MB) |
          |------------|-----------|
          | R80736.csv | 51.41     |
          | R80721.csv | 51.20     |
          | R80790.csv | 51.03     |
          | R80711.csv | 51.68     |
          
          | Library | Time (sec) | Max memory usage (MB) |
          |---------|------------|-----------------------|
          | pandas  | 6.13       | 735.23 MB             |
          | dask    | 18.38      | 603.20 MB             |
          | vaex    | 3.11       | 268.40 MB             |
          | modin   | 11.60      | 245.38 MB             |
                    
  2. Time Series and Forecast: learning from the past:
     
     - Calculates and visualises 3 quantities as function of time (Average Energy, Produced Energy and Capacity Factor)
     
     - Performs Auto-Regression analysis and Regularizing gradient boosting 

     ![plot](https://github.com/marcodigennaro/windml/blob/main/images/time_evolution_P_avg.jpeg)
     ![plot](https://github.com/marcodigennaro/windml/blob/main/images/monthly_evolution_P_avg.jpeg)
          
  3. Data Analysis

     - Extract, transform, and load (ETL) 
     - Exploratory Data Analysis (EDA) 
     - Feature Selection Analysis

     ![plot](https://github.com/marcodigennaro/windml/blob/main/images/windrose.jpeg)
     ![plot](https://github.com/marcodigennaro/windml/blob/main/images/heatmap.jpeg)
     

  4. Machine Learning

     - Perform several regression algorithms: Linear, Polynomial, Kernel Ridge Regression
     - Perform pipeline including grid search analysis for parameter optimization
     - Compare learning by plotting Learning Curve

     ![plot](https://github.com/marcodigennaro/windml/blob/main/images/scatter_plot.jpeg)
     ![plot](https://github.com/marcodigennaro/windml/blob/main/images/learning_curve.jpeg)


### Author

Marco Di Gennaro 
- [My GitHub](https://github.com/marcodigennaro)
- [My Linkedin](https://www.linkedin.com/in/marcodig/)
- [My professional website](https://atomistic-modelling.com/)

### License

This project is licensed under the GPL v3 License - see the [LICENSE.md](https://github.com/marcodigennaro/WindML/blob/main/LICENSE.md) file for details
 
### Acknowledgements

- A previous analysis on this database can be found [here](https://github.com/matteobonanomi/dsnd-wind-farm?tab=readme-ov-file)
- More on the XGBOOST algorithms can be found [here](https://www.youtube.com/watch?v=vV12dGe_Fho&t=1143s)



