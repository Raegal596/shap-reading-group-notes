# SHAP Reading Group Notes

Notes by David Reid [Raegal596](https://github.com/Raegal596)

This is a summary of the SHAP paper: A Unified Approach to Interpreting Model Predictions.

I recommend reading the author's GitHub page (<https://github.com/slundberg/shap>), before or instead of the paper. It provides examples of the applications of SHAP values. It also has pretty pictures and will make it easy to play around with SHAP values yourself.

## Overview

This paper proposes an explanatory model that can approximate linear regression-style coefficients from any predictive model. Building on previous work, it is both more rigourous and more computationally efficient than anything that has come before it.  The paper summarises previous explanatory models and shows that there they are connected. It then uses this link to combine them, generating a new explanatory model that they call SHAP values. SHAP values are as efficient as the cheapest of the previous models while providing explanations that are well aligned with the most rigourous and expensive one.

## Additive Feature Attribution Methods

The authors begin by defining a class of model explanations that they call additive feature attribution methods. These methods assign each feature a contribution value, which approximate the model's prediction when combined. You could think of contribution values as the coefficients in linear regression. Equation 1 emulates that.

They proceed to show that a whole series of model explanations belong to this class. The ones they mention in detail are LIME, DeepLIFT, and Shapley values (calculated in a variety of ways). They then provide a brief overview of each.

### LIME

LIME functions by assuming that the local parameter space around a data point is approximately linear. It then perturbs the parameters and feeds these altered data points through the predictive model.  The changes in model predictions are to fit a weighted linear regression model.

### DeepLIFT

DeepLIFT is a method that applies to neural networks. It calculates the difference between the predictions made for an input value and some reference value. A special case of DeepLIFT is Layer-Wise Relevance Propagation, where the reference values are all 0.

### Shapley Values

Shapley values calculate the contribution of each feature by replacing them out with a random value. They also consider the impact of switching features together, to account for interactions, and they do this for all possible higher-order interactions. The expected value of these combined effects is used to calculate as the contribution of the feature. This is the most rigourous of all the methods described, but it scales factorially with both the number of features and the size of the dataset. For that reason, the summation in the expectation value is usually truncated, and only a few permuted data points being used. They call this Shapley sampling.

## Simple Properties Uniquely Determine Additive Feature Attributions

This section focuses on some desirable properties that we'd like the interpretable model to have. Of the additive feature attribution methods mentioned, only Shapley values have all three of the following three properties:

1. **Local Accuracy** - the coefficients for the additive model should add up to the prediction made by the model.
2. **Missingness** - if a feature is absent from the input, then it's coefficient should be zero.
3. **Consistency** - if you change the model, and the new model predicts a higher value in all cases where a particular feature is present, then the coefficient for that feature should be larger, or at least not smaller.

The fact that Shapley values are the only explanation that guarantees these three properties was proved in 1985.

One of the reasons that these properties are helpful is you can now combine two models (for example (f + g) / 2) by combining their coefficients ((c_f + c_g) / 2). For example, in random forests, you could calculate the Shapley values for each tree and then add them together. This is something they don't mention in the paper, but it makes Shapley values much nicer to work with than the other methods they mention.

## SHAP (SHapley Additive exPlanations) Values

In this section, the authors define the SHAP values using a conditional expectation. This can be interpreted as the difference in the model prediction that could be expected if a feature was switched to something else. The expected value accounts for the fact that the feature could be switched to any number of things, and it could also be switched along with any number of other features. We're interested in is the average effect of making all those possible switches. This part is very similar to how Shapley values are defined.

They then define the SHAP kernel, which is a reformulated version of the LIME kernel, with the loss function and regularization terms replaced. They made these replacements in such a way that, in the limit of all possible permuted samples, the kernel would return the Shapley values. The nice thing about using the kernel rather than the expectation is that the weighted linear regression in LIME can be fit using fewer samples. This is faster to calculate than the traditional Shapley values.

While the SHAP kernel works for all models, for some you can speed up the calculation significantly if you tailor it to the specific architecture. The first three tweaks are basically: linear regression coefficients, kernel SHAP, and classic permutation SHAP. The fourth tweak is more novel. They define Deep SHAP, which is DeepLIFT reformulated so that the reference value is the average model prediction.

## Computation and User Study Experiments

After having defined everything, the authors decide to put their new algorithms to the test. The first test that they present compares LIME, the truncated Shapley values, and SHAP values to the exact Shapley values. Since these methods depend on repeatedly replacing values for features with others found in the dataset, their primary performance measure is the number of samples needed for SHAP values to converge to the true Shapley values. What they show is that the SHAP values converge faster than the Shapley sampling and about as soon as LIME. However, LIME values differ significantly from Shapley values.

They then examine the differences between the LIME and SHAP values in more detail and argue that the explanations that the SHAP values suggest are better aligned with what humans would come up with if they understood the underlying system that produced the data. They do this by getting people from Amazon Mechanical Turk to assign a weight to each feature for two simple systems and compare this to the values that both LIME and SHAP produce.

They also compare SHAP to DeepLIFT by looking at the change in feature attribution that occurs when you erase some of the pixels in MNIST to turn an 8 into a 3. Since the SHAP values are more sensitive and more closely resembles the Shapley values, they argue that they provide a better explanation for the decisions a neural network makes.