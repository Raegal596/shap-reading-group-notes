# SHAP Reading Group Notes
This is a summary of the SHAP paper: A Unified Approach to Interpreting Model Predictions.

Firstly, I highly recommend reading the author's GitHub page (https://github.com/slundberg/shap), possibly before (or instead of, if you're rushed) the paper. It provides some good examples of the applications of SHAP values, has a lot of pretty pictures, and will make it easy to play around with SHAP values yourself, which is the most interesting part.

I'll start with a brief overview for anyone who's pushed for time, then go into a section by section summary for anyone who isn't.

Briefly, this paper proposes a explanatory model that can be used to get linear regression style coefficients out of any model architecture. It does this in a way that is both more rigourous and more computationally efficient than anything that has come before it, though it borrows heavily from other work in order to do so. It summarises a number of explanatory models that have recently been proposed in the literature, and shows that there is a link connecting all of them. It then uses this link to combine them, generating a new explanatory model that they call SHAP values. These can be evaluated as quickly as the cheapest of the previous models, while providing explanations that are well alligned with the most rigourous and expensive one.

So ends the brief summary, and so begins the in depth one.

## Additive Feature Attribution Methods
The authors begin by defining a class of model explanations that they call additive feature attribution methods. These methods all assign each feature a contribution, which when added together approximate the model's prediction. The easiest way of thinking of this is in terms of the coefficients in linear regression, and the way they write equation 1 directly emulates that.

They then proceed to point out that a whole series of model explanations belong to this class. The ones they mention in detail are: LIME, DeepLIFT, and Shapley values (calculated in a variety of ways). They then provide a brief overview of each.

LIME functions by assuming that the local parameter space around a data point is approximately linear. It then permutes the parameters, feed these altered data points through the model, and use the changes in model predictions to fit a weighted linear regression model, with the weights being determined by how far (using some arbitrary distance metric) the permuted data points are from the original.

DeepLift is a method that applies specifically to neural networks and is similar to the backpropogation used to calculate gradients when training. It calculates the difference between the predictions made for a data point and some reference data point. It then uses a modified version of the chain rule to backpropogate this difference through the network, all the way to the input layer, where the values returned represent the feature's contribution to the prediction.

Shapley values come from game theory, and calculate the contribution of each feature by switching them out with some other random value. They also consider the impact of switching features together, in order to account for interactions, and they do this for all possible higher order interactions (3 features together, 4 features together, etc.). They take the expectation value of all of these, and use that as the contribution of the feature. This is the most rigourous of all the methods described, but it scales factorially with both the number of features, and the size of the dataset. For that reason, the summation in the expectation value is usually tuncated, and only a few permuted data points are usually used. This is what they call Shapley sampling.

## Simple Properties Uniquely Determine Additive Feature Attributions
This section focuses on some nice properties that we'd like the interpretable model to have. Of the additive feature attribution methods, as they formally defined them earlier, only one (spoiler alert, it's the Shapley values) have the following three properties:
1. **Local Accuracy** - the coefficients for the additive model should add up to the prediction made by the model.
2. **Missingness** - if a feature is absent from the input, then it's coefficient should be zero.
3. **Consistency** - if you change the model, and the new model predicts a higher value in all cases where a particular feature is present, then the coefficient for that feature should be larger, or at least not smaller.

The fact that Shapley values are the only explanation that guarantees these three properties was formally proved in 1985.

One of the reasons that these properties are nice is that they also mean that if you combine two models (for example (f + g) / 2), then you can also combine the coefficients in the same way ((c_f + c_g) / 2). This is important when it comes to implementation, since it means that for random forests, for example, you can calculate the coefficients for each tree individually, then add them together. This is something they don't mention in the paper, but it makes Shapley values much nicer to work with than the other methods they mention.

## SHAP (SHapley Additive exPlanations) Values
In this section they define the SHAP values using a conditional expectation. This can be interpreted as representing the difference in the model prediction that could be expected if a feature was switched to something else. The expectation value is there to account for the fact that the feature could be switched to any number of things, and it could also be switched along with any number of other features. What we're interested in is the average effect of making all those possible switches. This part is pretty much identical to the way Shapley values are defined.

They then define the SHAP kernel, which is a reformulated version of the LIME kernel, with the loss function and regularization terms replaced. They made these replacements in such a way that, in the limit of all possible permuted samples, the Kernel would return the Shapley values. The nice thing about using the kernel rather than the expectation value directly is that the weithged linear regression in LIME can be fit using far fewer samples, which makes it much faster to calculate than the classic Shapley values.

While the SHAP kernel works for all model architectures, for some you can speed up the calculation significantly if you tailor the it to the specific architecture. Although they discuss four methods, the first three are basically: linear regression coefficients, kernel SHAP, and classic permutation SHAP respectively. The fourth one is more novel. They define Deep SHAP, which is DeepLIFT reformulated (starting to see a pattern here) so that the reference value is the average model prediction. Aside from this, they borrow most of the formalism directly from DeepLIFT.

## Computation and User Study Experiments
After having defined everything, the authors decide to put their new algorithms to the test. The first test that they present compares LIME, the truncated Shapley values, and SHAP values to the exact Shapley values. Since all of these methods depend on repeatedly replacing values for features with others found in the dataset, their main measure of how well they perform is how quickly (in terms of number of samples) the explanations converge to the true Shapley values. What they show is that the SHAP values converge much more rapidly than the Shapley sampling, and that while LIME converges about as quickly, it converges to an explanation that differs significantly from the Shapley values.

They then examine the differences between the LIME and SHAP values in more detail, and argue that the explanations that the SHAP values suggest are better alligned with what humans would come up with if they understood the underlying system that produced the data. They do this by getting people from Amazon Mechanical Turk to assign a weight to each feature for two very simple systems, and compare this to the values that both LIME and SHAP produce.

They also compare SHAP to DeepLIFT by looking at the change in feature attribution that occurs when you erase some of the pixels in MNIST to turn an 8 into a 3. Since the SHAP values change more dramatically, and in a way that more closely resembles the Shapley values, they argue that they provide a better explanation for the decisions being made by the network.
