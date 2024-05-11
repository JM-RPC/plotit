This repository contains code for an early stage statistical visualization app using ShinyLive/Python. You can test it at https://jm-rpc.github.io/plotit/ . If you're here, I don't need to extoll the virtues of the Shiny Live deployment strategy. Fair warning: this code was thrown together quickly so it is buggy, use at your own risk. This project started as mostly a proof-of-concept exercise. And then got completely out of hand. Having said that, plotit offers a minimalist and fairly easy to use interface (well, easier than doing everything from the commandline), and will accomodate relatively large datasets( beyond about 2M rows it can be a bit slow). As far as possible, plotit repackages off-the-shelf functions from seaborn, statsmodels, spicy, etc. while at the same time trying to keep the number of required packages within reason (and within ShinyLive's limitations).
A design priority of plotIt was meant to be a simple, easy to use data exploration tool. It does not try to fit the data into a fixed format, but offers some standard statistical graphics for the user to try. Careful attention was paid to handling missing data. Rows with missing data (NaN's or na's in python) are not dropped on input. At any point where we need to drop rows containing missing data, only the rows that are missing data in the currently active columns (either being displayed or part of a linear model) will be dropped. This means that when you use a variable to color a plot there may be a NA catagory. You can remove it with the subsetting feature of plotIt.
Here's what plotit will do (so far):
1. Input tab: Open a .csv file of data from its local computing environment and give a simple summary of the data in the file. Currently, rows containing NaNs are dropped on input.
2. Correlations tab: Creates a grid of scatter plots and calculates Pearson correlations for chosen variables
3. Plots tab: User chooses variables (X, Y, Z, and a color variable), and type of plots: a. One variable: Histogram, Box Plot, KDE b. Two variables (x and y): 2D scatter plot with coloring. Three variables: interactive 3D plotting with plotly (am working on how to use rgl widgets to support matplotlib interactive 3D, so far no luck) Subsetting: For variables with fewer than 50 or so unique values, choose subsets based on the outcomes of the variables. You can subset on more than one variable.
4. Linear models: fit either a logistic or a linear model. If the dependent variable is binary, logistic regression is chosen automatically. Otherwise OLS. NOTE: the linear model tab will fit the model to the subset chosen in the Plots tab. Always check the number of observations!
5. After fitting a model, go back to the plotting page and choose Model Data  and you will be able to to explore your model, residuals, predictions etc.
6. Linear models: Standard Plots: a collection of standard plots for linear model diagnostics: ROC curve for logistic regression, leverage and influence for linear regression. Scatter against independent variables, etc.
To Do's

Data set splitting (training and test)

1. Poisson regression
2. Predictions using new data
3. Learn how to use Shiny modules
4. Maybe take requests for features......
5. As always, polite bug reports are appreciated. Suggestions for how to improve the code are greatly appreciated and will be cited if used. Gripes about my awful programming style are not appreciated (I already know this).
