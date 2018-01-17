# numer.ai

These are the codes I written for reaching 27th place (top 7%) in week 55 of [Numerai hedge fund competition](https://numer.ai/history/55).

This is a three-level stacking model. The first two levels are xgboost and lightgbm, while the last level is simple linear regression. All training and stacking are done with K-fold cross-validation to minimize information leakage. Feature engineering and parameter tuning are performed probabilistically through [Bayesian optimization](https://github.com/fmfn/BayesianOptimization/tree/master/bayes_opt).