# Classification Purpose

# Import necessary libraries
import pandas as pd
from sklearn.metrics import f1_macro
from sklearn.model_selection import cross_val_score


## Evaluate a single model -> f1_macro
def evaluate_single_model(model, xtrain, ytrain, xtest, ytest):
    # Cross validate the model on train data
    scores = cross_val_score(model, xtrain, ytrain, cv=5, scoring="f1_macro")
    # Calculate mean and std of scores
    cv_mean = scores.mean().round(4)
    cv_std = scores.std().round(4)
    # Fit the model
    model.fit(xtrain, ytrain)
    # Predict the results for train and test
    ypred_train = model.predict(xtrain)
    ypred_test = model.predict(xtest)
    # Calculate f1_macro for train and test
    f1_train = round(f1_macro(ytrain, ypred_train), 4)
    f1_test = round(f1_macro(ytest, ypred_test), 4)
    gen_err = round(abs(f1_train - f1_test), 4)
    # Provide results in dictionary
    res = {
        "name": type(model).__name__,
        "model": model,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "f1_train": f1_train,
        "f1_test": f1_test,
        "gen_err": gen_err,
    }
    return res


def algo_evaluation(models: list, xtrain, ytrain, xtest, ytest):
    # Define a blank results list
    results = []
    # Apply for loop on the model objects
    for model in models:
        r = evaluate_single_model(model, xtrain, ytrain, xtest, ytest)
        results.appnd(r)
        print(r)
    # Convert results to dataframe
    df_res = pd.DataFrame(results)
    # Sort as per cross validation score
    sort_res = df_res.sort_values(by="cv_mean", ascending=False).reset_index(drop=True)
    # Select the best model
    best_model = sort_res.loc[0, "model"]
    # Return below
    return sort_res, best_model
