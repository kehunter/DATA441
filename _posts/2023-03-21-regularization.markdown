---
layout: post
title:  "Regularization algorithms: Beyond Lasso and Ridge"
date:   2023-03-21 09:26:54 -0500
categories: projects
permalink: 
usemathjax: true
---

### Regularization

There are two main "actions" that you perform using a model in data science:
you **fit** the model using the training data that you have, and you **predict**
the outcomes of new data points (testing data in the case of validating a model)
using the fitted model.  However, some problems may spring up along each step of 
the process.  

Sometimes it is not possible to fit the model with the data that you have.  Fitting a regression model is often synonymous with solving some equation that includes your features ($X$).  Because $X$ can have more than one feature (i.e. multiple dimensions), linear algebra is necessary to solve some of these equations, including the equation for creating an Ordinary Least Squares model:

$$\large y = X\cdot\beta +\sigma\epsilon$$

Creating the model involves finding the $\beta$ coefficients for each feature.  Rearranging the equation with linear algebra (and ignoring the error term), we get: 

$$\large \beta = (X^TX)^{-1}(X^Ty)$$

So in order to create the model, we *need* to have $(X^TX)^{-1}$, and we know that it is only possible to invert a matrix if that matrix is *singular*
