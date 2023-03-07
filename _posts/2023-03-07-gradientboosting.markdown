---
layout: post
title:  "Gradient Boosting"
date:   2023-03-07 09:26:54 -0500
categories: projects
permalink: 
usemathjax: true
---

# Gradient Boosting

Gradient boosting is used to improve the predictions for new data, by using multiple regressors that each contribute a little bit to the final prediction.  Here is how it works: 
1. Fit the training data to the first regressor.
2. Make predictions for the training data using this regressor, and save the residuals.
3. Fit a new regressor with the *residuals* of the previous model.  
4. Repeat this fitting of new regressors if necessary or useful.  
5. Make predictions for new data using all fitted regressors, and add the predictions together for your final predictions. 

One way that I like to think of this process is allowing residuals of 
input data to "pull" the final model towards it.  Consider the graphic below:

![](../assets/gradientdescent/gradientdescent3.png)

Imagine that the residuals (red lines) of the fitted Regressor 1 are stuck to the model (dotted line).  Applying gradient descent is like
grabbing the dotted line at both ends and stretching it out to sit on the x-axis so that the training data points are the same distance from the x-axis as they were from Regressor 1.  Now a new regressor is fit to this new arrangement of the points that reflects the error
of the previous model.  Notice how data that had positive residuals in Regressor 1 will have positive predictions by Regressor 2, and negative residuals in Regressor 1 will have negative predictions by Regressor 2.  Once you add together the predictions from the two models (the blue dots), you can see how the residuals are essentially "pulling" the final predictions in their direction.  