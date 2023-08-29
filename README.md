# Module11_Project

## Summary Findings

### Business Understanding (BU)
  As per the overview, our business objective is to find and understand factors that would make a car more or less expensive.
  
  When putting it into data perspective, the factors are the columns in which we can also call them features and our targeted feature would be the column price.
  Below, we can consider all the non-price columns including the column 'id' not shown in the image as potential factors. 

  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/BU1.PNG)


### Data Understanding (DU)
  For the Data Understanding, I personally interpret what each column in the dataset represent.

  1. id - the label of the car
  2. region - the region where the car was manufactured (e.g. prescott, chico, little rock, stockton)
  3. price - how much the car cost
  4. year - when the car was manufactured
  5. manufacturer - what car company manufactured the car (e.g. gmc, chevrolet, toyota, ford, etc.)
  6. model - the version of the car (e.g. 96 Suburban, sierra 1500 crew cab slt, etc.)
  7. condition - the current state of the car
  8. cylinders - the amount of cylinders that the car's engine has
  9. fuel - the type of fuel the car relys on to run (e.g. gas, electric, etc.)
  10. odometer - the total distance that the car has run so far
  11. title_status - the legal ownership status of the car (e.g. lien, clean, missing)
  12. transmission - the type of mechanical system responsible for the car's transmission (e.g. manual, automatic, etc.)
  13. VIN - stands for Vehicle Identification Number which serves as a fingerprint for the vehicle
  14. drive - different types of drivetrains in cars that specify the car's wheels that the engine is distributing power to
  15. size - the size of the car (e.g. full-size, mid-size, etc.)
  16. type - the type of car it is (e.g. pickup, truck, bus, van, etc.)
  17. paint_color (e.g. white, blue, red)
  18. state - USA states (e.g. tx, az, ca)

  It's also easy to notice that the dataset has a lot of missing values.

  Here's the sum of null values in each column:
  
  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/DU1.PNG)
  
  with the DataFrame's shape (426880, 18) showing that some columns have a significant part of its data missing.
  
### Data Preparation (DP)
  For the Data Preparation, I first logically concluded that it wouldn't make any sense for the car's id and VIN to be able to associate with the car's price value and thus I dropped them.

  Because there are significant numbers of missing values, I decided to use an imputer to fill in missing values rather than to get rid of them all as that will sacrifice most of the data. I tried using various imputers like:
  #### KNNImputer
  ```python
  from sklearn.impute import KNNImputer
  ```
  #### KNN
  ```python
  from fancyimpute import KNN
  ```
  which required an installation into my Anaconda prompt using:
  ```python
  conda install -c conda-forge fancyimpute
  ```
  #### IterativeImputer
  ```python
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer
  ```
  #### fast_knn
  ```python
  from impyute.imputation.cs import fast_knn
  ```
  which required another installation but in my JupyterNotebook using:
  ```python
  pip install impyute
  ```
but all of them either cannot handle categorical data or required categorical encoding in which when input in the imputer, the imputer will run for a long time.

Thus, I stuck with the 
#### SimpleImputer 
```python
from sklearn.impute import SimpleImputer
```
whose hyperparameter I set to 'most_frequent' meaning all missing values shall be replaced by the frequent values of its respective column.

I also noticed that some price values are 0 which didn't make any sense because cars including used ones aren't supposed to be free so I replaced them all with the median price value of all nonzero price values.

This is the final dataset I got before I started modeling:
![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/DP1.PNG)

### Modeling (Mo)
  Before modeling, I of course train_test_split my data with hyperparameter test_size set to 0.3, i.e. 0.7:0.3 (Train:Test) and hyperparameter random_state set to 1.
  I then categorically encoded the final dataset with the CatBoostEncoder after importing from:
  ```python
  from category_encoders import CatBoostEncoder
  ```
  in which I need to personally install in my Anaconda prompt using:
  ```python
  conda install -c conda-forge category_encoders
  ```
  It turns out I then decided to use this as a final encoded dataset for all 5 models that I'm about to experiment on. They are:
  #### CatBoostRegressor
  ```python
  from catboost import CatBoostRegressor
  ```
  in which I need to install in JupyterNotebook using:
  ```python
  pip install catboost
  ```
  with its Root-Squared-Mean-Error (RMSE) and Mean-Absolute-Error (MAE) being
  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Mo1.PNG)
  

### Evaluation

### Deployment
