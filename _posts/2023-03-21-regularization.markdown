---
layout: post
title:  "Regularization"
date:   2023-03-21 09:26:54 -0500
categories: projects
permalink: 
usemathjax: true
---

## **What is Regularization?**
Regularization is a technique that causes a model to be less sensitive to variations in the training data. It does so by minimizing the sum of a loss function (which is typically minimized when fitting a model) and a penalty function.  For example, a regression model may minimize the mean squared error but a regularized regressor would minimize the mean squared error plus some penalty on the $$\beta$$ coefficients controlled by some hyperparameter $$\alpha$$.  This helps solve problems with overfitting and linearly dependent data. 

The following two equations are minimized in regularization, where $$\alpha$$ is a hyperparameter that determines how much of a penalty the coefficients will carry:

L1 regularization (Lasso): 

$$\frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2 + \alpha\sum_{j=1}^p|\beta_j|$$

L2 regularization (Ridge):

$$\frac{1}{n}\sum_{i=1}^n(y_i - \hat{y}_i)^2 + \alpha\sum_{j=1}^p\beta_j^2$$


### **When do I need regularization?**

In data science, once a model is chosen for a dataset, there are two main "actions" that are necessary to put the model into use:
the model is $$\textcolor{blue}{fit}$$ to training data, and the fitted model is used to $$\textcolor{orange}{predict}$$ outcomes of new or test data points.  
![]({{site.baseurl}}/assets/images/regularization/img_linear_regression2.png)

Image source: [W3Schools](https://www.w3schools.com/python/python_ml_linear_regression.asp)

However, some problems may arise along each step of the process that we can solve using **regularization**. 

#### **Problem 1: linearly dependent data causes trouble *fitting* model**

Sometimes it is not possible to fit the model with the data that you have.  Fitting a regression model is often synonymous with solving some equation that includes your features ($$X$$).  Because $$X$$ can have more than one feature (i.e. multiple dimensions), linear algebra is necessary to solve some of these equations, including the equation for creating an Ordinary Least Squares model:

$$\large y = X\cdot\beta +\sigma\epsilon$$

Creating the model involves finding the $$\beta$$ coefficients for each feature.  Rearranging the equation with linear algebra (and ignoring the error term), we get: 

$$\large \beta = (X^TX)^{-1}(X^Ty)$$

So in order to create the model, we *need* to have $$(X^TX)^{-1}$$, and to get this, $$(X^TX)$$ must be *invertible*.  If a square matrix is *invertible*, it is *non-singular*, with a nonzero determinant, and linearly independent columns.  

**This means that if our dataset produces a square $$X^TX$$ matrix that is *not* invertible, then it is impossible to solve for the coefficients of the model and OLS cannot be used.**

One way in which the matrix of features, $$X$$, can lead to a non-invertible matrix is if the number of observations, $$n$$, is lower than the number of features, $$p$$.  For example, genomic datasets can contain thousands of features representing the expression of each gene in a sample, but only contain hundreds of observations or less, due to how expensive it is to sequence these samples.  This dataset (size: $$[8,2000]$$) could not be fit to an Ordinary Least Squares model: 

![]({{site.baseurl}}/assets/images/regularization/genetic_table_eg.png)

Regularization solves this issue by causing a slight perturbation in the matrix of features in order to make them linearly independent. 


#### **Problem 2: overfitting causes trouble *predicting* the outcome**

In other cases, it is possible to fit the training data to the model, but it is tuned so well to the training data that it performs very poorly when predicting values for new observations. In the image below, notice how the black line in the image below is representative of the roughly linear trend of the data, while the blue line shows an unrealistic trend despite fitting all of the data perfectly. The blue line shows a model that is **overfit**. 

![]({{site.baseurl}}/assets/images/regularization/Overfitted_Data.png)

Source: [Wikipedia](https://en.wikipedia.org/wiki/Overfitting)

Regularization solves this issue by giving less weight to each feature and "smoothing" the model. 

![]({{site.baseurl}}/assets/images/regularization/regularized.png)

Source: [MathWorks](https://www.mathworks.com/discovery/regularization.html)


## Square Root Lasso and Smoothly Clipped Absolute Deviations (SCAD)

### Scikit-Learn Implementation 

### Tuning the hyperparameters

### Apply to air quality dataset

The source for the air quality dataset:
Chambers, JM, Cleveland, WS, Kleiner, B & Tukey, PA (1983), Graphical Methods for Data Analysis, Wadsworth, Belmont, CA.