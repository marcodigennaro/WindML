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
Data should be downloaded beforehand. 

### Content of the Jupyter Notebooks

  1. Scalability

     - Tests the memory/speed performances of 4 python libraries

  2. Time Series and Forecast: learning from the past:
     
     - Calculates and visualises 3 quantities as function of time (Average Energy, Produced Energy and Capacity Factor)
     
     - Performs Auto-Regression analysis and Regularizing gradient boosting 
     
  3. Data Analysis

     - Extract, transform, and load (ETL) 
     - Exploratory Data Analysis (EDA) 
     - Feature Selection Analysis
     
  4. Machine Learning

     - Perform several regression algorithms on the data and compare them with Learning Curve


### Data

Data are available at this [URL](https://opendata-renewables.engie.com/pages/home/).
Since this is not always functioning, a `data` folder was included in this package.

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



