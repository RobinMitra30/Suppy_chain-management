# Suppy Chain Management
This exercise aims to build a model, using historical data that will determine the optimum weight of the product to be shipped each time to the warehouse.

## Approach:
Loading the dataset using Pandas and performing basic checks like the data type of each column and any missing values.
Performed Exploratory data analysis:
Visualized each predictor or independent feature with the target feature and found that there's a direct proportionality between cement and the target feature while there's an inverse proportionality between water and the target feature.
To get even more better insights, plotted both Pearson and Spearman correlations, which showed the same results as above.
The distribution of the target feature, expenses was in Normal distribution with a very little right skewness.
Checked for the presence of outliers in all the columns

Experimenting with various ML algorithms:
First, tried Linear regression models. Performance metrics are calculated for the approach. The test RMSE score is a little bit lesser compared to other approaches. Then, performed a residual analysis and the model satisfied all the assumptions of linear regression.
Next, tried various tree-based models. We use a voting algorithm which helps us to identify which algorithm is best. Then, pick the top features as per the feature importance of each model. Models, evaluated on both the training and testing data and recorded the performance metrics.
Based on the performance metrics of both the linear and the tree-based models, the decision tree regressor performed the best, followed by the random forest regressor. 

## Deployment: 
Deployed the Gradient Boosting regressor model using Flask, which works in the backend part while for the frontend UI Web page, used HTML5.


So, now we can find the insurance premium quickly by just passing the mentioned details as input to the web application 
