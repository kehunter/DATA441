---
layout: post
title:  "Gradient Boosting"
date:   2023-03-07 09:26:54 -0500
categories: projects
permalink: 
usemathjax: true
---

### Background

Gradient boosting is a type of ensemble method that is used to improve the predictions for new data, by using multiple regressors that each contribute a little bit to the final prediction.  Here is how it works: 
1. Fit the training data to the first regressor.
2. Make predictions for the training data using this regressor, and save the residuals.
3. Fit a new regressor to the *residuals* of the previous model.  
4. Repeat this fitting of new regressors if necessary or useful.  
5. Make predictions for new data using all fitted regressors, and add the predictions together for your final predictions. 

One way that I like to think of this process is that you are allowing the residuals 
of the previous model to "pull" the final model towards it.  Consider the graphic below:

![]({{site.baseurl}}/assets/images/gradientdescent/gradientdescent4.png)

Imagine that the residuals (red lines) of the fitted Regressor 1 are stuck to the model (dotted line).  Applying gradient descent is like
grabbing the dotted line at both ends and stretching it out to sit on the x-axis so that the training data points are the same distance from the x-axis as they were from Regressor 1.  Now a new regressor is fit to this new arrangement of the points that reflects the error
of the previous model.  Notice how data that had positive residuals in Regressor 1 will have positive predictions by Regressor 2, and negative residuals in Regressor 1 will have negative predictions by Regressor 2.  Once you add together the predictions from the two models (the blue dots), you can see how the residuals are essentially "pulling" the final predictions in their direction.  The animation found on the [University of
Cincinnati Business Analytics Programming guide](http://uc-r.github.io/gbm_regression) provides a great visual of this:

![]({{site.baseurl}}/assets/Animation/boosted_stumps.gif)

Typically, this method is used with classifiers over regressors, because it would not make sense 
to perform gradient boosting using simple linear regression models.  For example, Ordinary Least Squares (OLS) linear regression works under the assumption that the mean of the residuals is 0 and that the residuals are identically and independently distributed. If a second linear regression line were fit to a plot of the residuals, you would expect the model to be identical to the x-axis, which obviously does not help your predictions. 


### My implementation

Creating a function to perform gradient boosting is very simple:

{% highlight python %}
def gradient_boosting(X, y, Xnew, regressor1, regressor2):
  # fit the first regressor
  regressor1.fit(X,y)
  # get the residuals
  residuals = y - regressor1.predict(X).reshape(-1,1) 
  # get predictions for the new data
  pred1 = regressor1.predict(Xnew)
  # fit the second regressor on the residuals
  regressor2.fit(X,residuals)
  # combine the predictions from both regressors
  return pred1 + regressor2.predict(Xnew)
{% endhighlight %}

As a continuation of my exploration of locally 
weighted linear regression (LOWESS), I will test this function with LOWESS regressors (I made minor modifications to a LOWESS regressor function to add the option to specify the kernel), as well
as Random Forest regressors.  

For the sake of example, I am not going into the details of the datasets that I
am working with. 

I implemented a KFold Cross-validation: 

{% highlight python %}
# storage for how well regressors perform
mse_lws = []
mse_rf = []

# regressors to test 
model_lws = Lowess_AG_MD(kernel = Gaussian)
model_rf = RandomForestRegressor(n_estimators=200,max_depth=5)

# kfold crossvalidation
kf = KFold(n_splits = 10, shuffle = True, random_state = 99)
ss = StandardScaler()
for idxtrain, idxtest in kf.split(X): # for each of ten splits
  # split the data
  Xtrain = X[idxtrain]
  Xtest = X[idxtest]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  # scale the data
  Xtrain = ss.fit_transform(Xtrain)
  Xtest = ss.transform(Xtest)
  # perform gradient boosting
  lws_pred = gradient_boosting(Xtrain, ytrain, Xtest, model_lws, model_lws)
  mse_lws.append(mse(ytest, lws_pred))
  rf_pred = gradient_boosting(Xtrain, ytrain, Xtest, model_rf, model_rf)
  mse_rf.append(mse(ytest, rf_pred))
{% endhighlight %}

### Testing: cars dataset

First, let's take a look at a dataset representing 392 observations of different cars:

<img src="{{site.baseurl}}/assets/images/gradientdescent/carsrs.head.png" height = "300" width = "300">

{% highlight python %}
X = cars[["CYL","ENG","WGT"]].values
y = cars["MPG"].values
{% endhighlight %}

Running the KFold:
{% highlight python %}
The Cross-Validated Mean Square Error for LOWESS is:  18.321989232951616
The Cross-Validated Mean Square Error for Random Forest is:  17.316067076546528
{% endhighlight %}

Random forest slighly outperformed LOWESS when both were performed with gradient boosting. 
Let's see how the mean square errors look when we compare them to models that did
not undergo gradient boosting:

{% highlight python %}
The Cross-Validated MSE for LOWESS (no gradient boosting):  18.58358030307805
The Cross-Validated MSE for Random Forest (no gradient boosting):  17.690344235491086
{% endhighlight %}

As expected, the mean squared errors for the testing data were slightly higher
for the models when no gradient boosting was performed.  This reveals how gradient 
boosting may be useful for improving models on new data.

### Testing: concrete dataset

The next dataset contains 1030 observations for different concretes:
<img src="{{site.baseurl}}/assets/images/gradientdescent/concrete.head.png" height = "300" width = "700">

I am going to predict concrete strength using 8 features:

{% highlight python %}
X = concrete.loc[:,"cement":"age"].values
y = concrete["strength"].values
{% endhighlight %}

This took around an hour to run, so it might be better to run gradient boosting KFold validation on fewer PCAs in application.  Random forest clearly seems to be a better model to use in this case.  

{% highlight python %}
The Cross-Validated Mean Square Error for LOWESS is:  114.25041569434867
The Cross-Validated Mean Square Error for Random Forest is:  30.932740179473363
{% endhighlight %}

Using a Random Forest model *without* gradient boosting does not perform as well on the testing data:

{% highlight python %}
The Cross-Validated MSE for Random Forest (no gradient boosting):  45.152328541144236
{% endhighlight %}

### Testing: housing dataset

Finally, I will perform one more cross-validation on a dataset representing 506 houses:
<img src="{{site.baseurl}}/assets/images/gradientdescent/housing.head2.png" height = "300" width = "600">

Let's see how well the features are predictive of median house value:

{% highlight python %}
X = housing.loc[:,["industrial","nox","older"]].values
y = housing["cmedv"].values
{% endhighlight %}

{% highlight python %}
The Cross-Validated Mean Square Error for LOWESS is:  64.9252649853525
The Cross-Validated Mean Square Error for Random Forest is:  39.21292901951854
{% endhighlight %}

Again, the MSE with gradient boosting is better, but only slightly:

{% highlight python %}
The Cross-Validated MSE for LOWESS (no gradient boosting):  66.18413935372048
The Cross-Validated MSE for Random Forest (no gradient boosting):  41.04994943197759
{% endhighlight %}


### Testing other kernels

To improve the model for a given dataset, we can use different options of kernels
within the LOWESS function.
I tested the Gaussian, Tricubic, Quartic, and Epanechnikov kernels for the housing 
dataset and got the following MSEs:

{% highlight python %}
X = housing.loc[:,["industrial","nox","older"]].values
y = housing["cmedv"].values
TTS = tts(X,y)
Xtrain = TTS[0]
Xtest = TTS[1]
ytrain = TTS[2]
ytest = TTS[3]
results = dict()

kernels = [Gaussian, Epanechnikov,Tricubic, Quartic]
for kernel in kernels:
  model = Lowess_AG_MD(kernel = kernel)
  yhat = gradient_boosting(Xtrain, ytrain, Xtest, model, model)
  results[str(kernel)] = mse(ytest, yhat)
results
{% endhighlight %}

{% highlight python %}
Boosted MSE when LOWESS used Gaussian kernel: 74.3340926299202
Boosted MSE when LOWESS used Epanechnikov kernel: 67.28711432066056
Boosted MSE when LOWESS used Tricubic kernel: 67.08200069052096
Boosted MSE when LOWESS used Quartic kernel: 66.76832683122684
{% endhighlight %}


As you can see, the Quartic kernel appeared to produce the best
predictions for the testing data on one train/test split. 

## Summary

In conclusion, gradient boosting appears to improve the mean squared
error of models - even when boosted only by one additional regressor -
 by pulling the final model in the direction of the
residuals from previous model fits.  