# Intro to Locally Weighted Linear Regression

## What is Locally Weighted Linear Regression?

### "Linear Regression" 

Linear regression is a modeling technique that uses linear combinations of the
input data to give a prediction of the dependent variable's value for a given point.
The general form of the equation that defines a linear regression model is as follows: 

$$\large y = X\cdot\beta +\sigma\epsilon $$

where 
- $y \rightarrow$ vector of the dependent variable values
- $X \rightarrow$ matrix of observations
- $\beta \rightarrow$  a vector of coefficients calculated by minimizing the SSE of the observed y values
from the predicted y values
- $\epsilon \rightarrow$ random error 


In order to calculate the $\beta$ coefficients, we must solve for $\beta$. To get our 
matrix of features on the other side of the equation, we will have to multiply by its 
inverse which is only possible if the matrix is invertible, and necessarily square. 
Multiplying X by its transpose will make it into a (likely) invertible square matrix: 

$$\large  X^Ty = X^TX\beta +\sigma X^T\epsilon$$

And after checking that $X^TX$ is invertible, we can solve for $\beta$:

$$\large \beta = (X^TX)^{-1}(X^Ty) - \sigma (X^TX)^{-1}X^T\epsilon$$

We can never truly know the random error term, so we take an estimate:

$$\large \bar{\beta} = (X^TX)^{-1}(X^Ty)$$

Finally, we can plug in our estimated $\beta$ coefficients and obtain 
our linear regression equation to use for predictions:

$$\large \hat{y} = X(X^TX)^{-1}(X^Ty)$$

Linear regression can be a great method for modeling and predicting linear data, but 
it cannot directly be applied to non-linear trends, like in the image below:

![](https://beginningwithml.files.wordpress.com/2018/07/2-e1530546876638.png)

Image credit to Rahul Yedida (2018). 

### "Locally"

If you tried to draw a single line through that data, the predictions would be very poor. 
However, if you break the data up into smaller segments/datasets, you can apply 
linear regression to these miniature datasets and draw a linear regression line for each one.
This calculus-like approach approximates the true curve of the line with multiple small, 
linear models. This way, when you are making a prediction for a new observation, you can 
localize the problem to a smaller neighborhood of the data: 

![](https://beginningwithml.files.wordpress.com/2018/07/3.png)

Image credit to Rahul Yedida (2018).

### "Weighted" 

So how do we localize the data? We use what is known as a **kernel function** which assigns each
data point a **weight** based on how far that point is from a central point. The kernel looks like 
a bell-shaped curve, so that points that fall outside of the scope of the kernel receive a value of 0.
For LOWESS, each point that we are trying to predict will be the center point for the kernel, giving
us a matrix of weights. 

The kernel width is defined by a hyperparameter called tau, $\tau$:

1. Theory and math
2. Kernel Functions
3. Ways of coding function
4. Plots

![](https://miro.medium.com/max/1400/1*H3QS05Q1GJtY-tiBL00iug.webp)
Image credit to Suraj Verma.
