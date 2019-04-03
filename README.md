# SHAP Reading Group Notes
Summary of the SHAP paper: A Unified Approach to Interpreting Model Predictions.

Firstly, I highly recommend reading the author's (GitHub page)[https://github.com/slundberg/shap], possibly before (or instead of, if you're rushed) the paper. It provides some good examples of the applications of SHAP values, has a lot of pretty pictures, and will make it easy to play around with SHAP values yourself, which is the most interesting part.

## Additive Feature Attribution Methods

## Simple Properties Uniquely Determine Additive Feature Attributions
This section focuses on some nice properties that we'd like the interpretable model to have. For the additive feature attribution methods, as they formally defined them earlier, there is only one (known as the Shapley values) have the following three properties:
1. **Local Accuracy** - the coefficients for the additive model should add up to the prediction made by the model.
2. **Missingness** - if a feature absent in the input, then it's coefficient should be zero.
3. **Consistency** - if you change the model, and the new model predicts a higher value in all cases where a particular feature is present, then the coefficient for that feature should be larger, or at least not smaller.

This was formally proved in game theory in 1985.

One of the reasons these properties are nice is that, they also mean that if you combine two models (f + g) / 2, then you can also combine the coefficients in the same way (c_f + c_g) / 2. This is important when it comes to the implementation side of things, since it means that for random forests, for example, you can calculate the coefficients for each tree individually, then add them together. This is something they don't mention in the paper, but it makes Shapley values much nicer to work with than the other methods they mention.

## SHAP (SHapley Additive exPlanations) Values
In this section they define the SHAP values using a conditional expectation. This can be interpreted as the difference it would make to the model prediction that could be expected if a feature was switched to something else. The expectation value is there to account for the fact that the feature could be switched to any one of a number of things, and it could also be switched along with any number of other features. What we're interested in is the average effect of making all those possible switches.

They then define the SHAP kernel, which is a reformulated version of the LIME kernel, with the loss function and regularization terms replaced with values that give it the properties they talked about in the previous section. The nice thing about this kernel, is that it is a locally linear approximation, which makes it much faster to calculate than making all the variable permutations that would be necessary to calculate the Shapley values the classical way.

While the SHAP kernel works for all model architectures, for some you can calculate the SHAP values faster if you tailor the calculation to the specific architecture. Although they discuss four, the first three are basically: linear regression coefficients, kernel SHAP, and classic permutation SHAP respectively. The fourth one is more novel. They define Deep SHAP, which is DeepLIFT reformulated (starting to see a pattern here) so that the reference value is the average model prediction. Aside from this, they borrow most of the formalism directly from DeepLIFT.

## Computation and User Study Experiments
