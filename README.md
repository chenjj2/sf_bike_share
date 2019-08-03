# SF Bay Area Bike Share Trips

## Data
`https://www.kaggle.com/benhamner/sf-bay-area-bike-share`

## Problem
Predict the time a trip will take, 
given the start and end bike station.

## Method

#### Data Inclusion Criteria
We filtered out instances that satisfies one of the following:
1. trips with same start and end station
2. trips across cities
3. mean velocity [*] below 5km/h or higher than 30km/h

[*] mean velocity here is defined as the L2-distance divided by duration.

#### Sub-Sampling
We sampled 10,000 instances from the original data for training and testing (9 : 1).

#### Feature Engineering
The following features are included/constructed:
1. day of week
2. subscription type
3. latitude/longitude of start and end
4. L2-distance between start and end
5. city
6. if it is peak hour (8, 9, 16, 17, 18)
7. mean temperature, dew point, humidity, pressure, visibility, wind speed
8. maximum gust speed
9. precipitation
10. cloud cover

#### Model
We tried a variety of methods in the sklearn package, including linear regression, LASSO, ridge, gradient boosting, and neural network.
Then we fine tuned parameters on several models.

## Performance
The best neural network structure is with (50, 20, 3)-RELU hidden layers, giving R^2 = 0.61 and MSE = 32321 (about 3 min error).

The best gradient boosting regression is with maximum depth = 3, giving R^2 = 0.69 and MSE = 25500 (2 min and 40s error).
