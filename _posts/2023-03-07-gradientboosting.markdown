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