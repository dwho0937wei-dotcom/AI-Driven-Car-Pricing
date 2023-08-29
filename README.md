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
  
  and its feature importance being

  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Mo2.PNG)

  #### Ridge Regression Model
  ```python
  from sklearn.linear_model import Ridge
  ```
  with RMSE and MAE being
  
  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Mo3.PNG)

  and its feature importance being

  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Mo4.PNG)

  #### Lasso Regression Model
  ```python
  from sklearn.linear_model import Lasso
  ```
  with RMSE and MAE being

  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Mo5.PNG)  

  and its feature importance being

  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Mo6.PNG)

  #### Random Forest Regressor
  ```python
  from sklearn.ensemble import RandomForestRegressor
  ```
  with RMSE and MAE being

  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Mo7.PNG)

  and its feature importance being

  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Mo8.PNG)

  #### XGBoost Model
  ```python
  import xgboost as xgb
  ```
  which required an installation on JupyterNotebook using:
  ```python
  pip install xgboost
  ```
  with RMSE and MAE being

  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Mo9.PNG)

  and its feature importance being

  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Mo10.PNG)

### Evaluation (Ev)
  After comparing all the RMSES and MAES of all five models, I came to the conclusion that the Random Forest Regressor will be the best model because it has the lowest MAE compared to the other 4 models.
  I then rely on its feature importance

  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Mo8.PNG)

  to conclude that the paint_color, condition, and year are the top three features that are the most impactful to the price of the car.

  However, the feature importance cannot tell us on whether they increase or decrease the price of the car. Thus, I relyed on scatterplot and heatmap for each feature. For the sake of not overwhelming the heatmap with numerous different price values, I've assigned all the price values to either Low, Medium-Low, Medium-High, or High in which they represent the four equidistant ranges of all different numeric price values. Note that none of the price values were assigned to Medium-High as the heatmaps below will imply.

  For Paint Color, we have
  
  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Ev1.PNG)
  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Ev2.PNG)

  showing that cars that were either colored green, silver, or white have the possibility of being highly priced. The black cars only have the possibility of being medium-low priced at max with all the other unspecified colored cars being only on the low priced range.

  For Year, we have

  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Ev3.PNG)
  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Ev4.PNG)  

  showing that the more recent the cars have been manufactured, the more possibility they have of being highly priced.

  For Condition, we have

  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Ev5.PNG)
  ![alt text](https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/images/Ev6.PNG)    

  showing that cars in either fair, good, or excellent condition has the possibility of being highly priced although the majority of them are still low priced. Still better than the other conditions in which all of them are low priced.

  #### Business or Data Issues
Keep in mind that before the models were used, many categorical NaN values were replaced by frequent values of its respective column and the majority of prices that were initially 0 were changed into median prices. Thus, it can be said that the same-value presence in the modified car dataset is significant enough to make the dataset inaccurate, bias, and/or less credible.

There may be other ways to fix these very noticeable issues but unfortunately, I no longer have the time to find any. Thus, we are moving on to the Deployment with the Random Forest Regressor remaining as the best model.


### Deployment
#### How will the model work when deployed

To start off, this model can be deployed as a form of the car dealers' inventory that they can input car datasets  of their own with similar structures as my given one, i.e. each row represents an individual car and there can be  numerous features about the car in which one of the features has to be the price of the car in order for this model to work.

It is highly recommended that the dataset has no missing values nor values that won't make any sense in a real-life scenario. Otherwise, if the model detect any missing/nonsensical values, then the model will automatically fill    them in with frequent values for categorical features or with a median value for numeric features using            SimpleImputer.

The model will use CatBoost Encoder to encode the dataset and before using its best model Random Forest Regressor, it will ask the car dealer to choose a number from 2 to 12 telling the dealer that it represents a range from fast but probably inaccurate results to more slow but more accurate results. After the car dealer chose the number, that number will be input into the hyperparameter n_estimators of the Random Forest Regressor.

Once these three features are found, it will then create a scatterplot and a heatmap for each feature for the car  dealer to see. It will interpret what the scatterplot and the heatmap meant for that specific feature relating to  price. Also, to avoid the heatmap from being overwhelmed with numerous different numeric values, the model will    automatically use np.linspace and pd.cut to create a new column with four categorical bins covering four equal     ranges of all different numeric values to have these four be the labels for the heatmap instead of the overwhelming different numeric values.

#### My findings interpreted to clients

Before delivering my findings to the car dealers, I will first let them know that there were many                  missing/nonsensical values in my dataset which are replaced by the most frequent or median values of its respective feature so my findings may be bias to that regard.

According to my model, the paint color of the car should be the first factor they should keep in mind when it comes to the price of the car. I will specifically tell them that there's a slim chance that consumers highly value      green, silver, and white cars with black cars being the secondary best. The rest of the colors seemed to be          completely lowly valued according to its price.

Next up would be the year that the car was manufactured. It would seem that the more recent the car has been       manufactured, then the more chances the car will have to be highly valued or have a high price. Thus, if you want  used cars that are highly valued, look for less old-fashioned and more recent trendy ones.

The last is the condition of the car. I cannot explain to them why cars that are in like-new or new conditions are not highly valued but I will sure tell them to also avoid ones in salvage condition, i.e. used cars that are no    longer roadworthy and cannot be repaired. Used cars that are either in fair, good, or excellent conditions however are what consumers highly valued. Thus, make sure to look after your used cars if you want your consumers to buy    them at a high price.


#### My Relevant JupyterNotebook
https://github.com/dwho0937wei-dotcom/Module11_Project/blob/main/prompt_II_Daniel-Ho.ipynb 
