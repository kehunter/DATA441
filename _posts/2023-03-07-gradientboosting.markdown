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

One way that I like to think of this process is allowing residuals of 
input data to "pull" the final model towards it.  Consider the graphic below:

![]({{site.baseurl}}/assets/images/gradientdescent/gradientdescent3.png)

Imagine that the residuals (red lines) of the fitted Regressor 1 are stuck to the model (dotted line).  Applying gradient descent is like
grabbing the dotted line at both ends and stretching it out to sit on the x-axis so that the training data points are the same distance from the x-axis as they were from Regressor 1.  Now a new regressor is fit to this new arrangement of the points that reflects the error
of the previous model.  Notice how data that had positive residuals in Regressor 1 will have positive predictions by Regressor 2, and negative residuals in Regressor 1 will have negative predictions by Regressor 2.  Once you add together the predictions from the two models (the blue dots), you can see how the residuals are essentially "pulling" the final predictions in their direction.  

Typically, this method is used with classifiers over regressors, because it would not make sense 
to perform gradient boosting using simple linear regression models.  For example, Ordinary Least Squares (OLS) linear regression works under the assumption that the mean of the residuals is 0 and that the residuals are identically and independently distributed. If a second linear regression line were fit to a plot of the residuals, you would expect the model to be identical to the x-axis, which obviously does not help your predictions. 

### My implementation

Creating a function to perform gradient boosting is in and of itself very simple:

{% highlight python %}
def gradient_boosting(X, y, Xnew, regressor1, regressor2):
  regressor1.fit(X,y)
  residuals = y - regressor1.predict(X).reshape(-1,1) 
  pred1 = regressor1.predict(Xnew)
  regressor2.fit(X,residuals)
  return pred1 + regressor2.predict(Xnew)
{% endhighlight %}

As a continuation of my exploration of locally 
weighted linear regression (LOWESS), I will test this function with LOWESS regressors (I made minor modifications to a LOWESS regressor function to add the option to specify the kernel).  

Here is the code for the KFold Crossvalidation that I implemented: 

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



### Testing: concrete dataset

The next dataset contains 1030 observations for different concretes:
<img src="{{site.baseurl}}/assets/images/gradientdescent/concrete.head.png" height = "300" width = "700">



{% highlight python %}
X = concrete.loc[:,"cement":"age"].values
y = concrete["strength"].values
{% endhighlight %}

This took around an hour to run, so it practically might be better to run gradient boosting KFold validation on fewer PCAs in application.  

{% highlight python %}
The Cross-Validated Mean Square Error for LOWESS is:  114.25041569434867
The Cross-Validated Mean Square Error for Random Forest is:  30.932740179473363
{% endhighlight %}

### Testing: housing dataset

Finally, I will perform one more cross-validation on a dataset representing 506 houses:
<img src="{{site.baseurl}}/assets/images/gradientdescent/housing.head.png" height = "300" width = "750">

{% highlight python %}
X = housing.loc[:,["industrial","nox","older"]].values
y = housing["crime"].values
{% endhighlight %}

{% highlight python %}
The Cross-Validated Mean Square Error for LOWESS is:  62.93571867291489
The Cross-Validated Mean Square Error for Random Forest is:  43.940105754548306
{% endhighlight %}


### Testing other kernels

To improve the model for a given dataset, we can use different options of kernels
within the LOWESS function.
I tested the Gaussian, Tricubic, Quartic, and Epanechnikov kernels for the housing 
dataset and got the following MSEs:

```
X = housing.loc[:,["industrial","nox","older"]].values
y = housing["crime"].values
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

{'<function Gaussian at 0x7f0cd35cd280>': 98.11231538326136,
 '<function Epanechnikov at 0x7f0cd35cdc10>': 76.79082033945697,
 '<function Tricubic at 0x7f0cd35cdaf0>': 76.84598324759428,
 '<function Quartic at 0x7f0cd35cdb80>': 74.10630540196719}
```

As you can see, the Quartic kernel appeared to produce the best
predictions for the testing data on one train/test split. 