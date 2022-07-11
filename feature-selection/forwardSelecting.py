import pandas as pd
import statsmodels.api as sm

from utils import __varcharProcessing__


def __forwardSelectionRaw__(X, y, model_type="linear", elimination_criteria="aic", sl=0.05):
    iterations_log = ""
    cols = X.columns.tolist()

    def regressor(y, X, model_type=model_type):
        if model_type == "linear":
            regressor = sm.OLS(y, X).fit()
        elif model_type == "logistic":
            regressor = sm.Logit(y, X).fit()
        else:
            print("\nWrong Model Type : " + model_type + "\nLinear model type is seleted.")
            model_type = "linear"
            regressor = sm.OLS(y, X).fit()
        return regressor

    selected_cols = ["intercept"]
    other_cols = cols.copy()
    other_cols.remove("intercept")

    model = regressor(y, X[selected_cols])

    if elimination_criteria == "aic":
        criteria = model.aic
    elif elimination_criteria == "bic":
        criteria = model.bic
    elif elimination_criteria == "r2" and model_type == "linear":
        criteria = model.rsquared
    elif elimination_criteria == "adjr2" and model_type == "linear":
        criteria = model.rsquared_adj

    for i in range(X.shape[1]):
        pvals = pd.DataFrame(columns=["Cols", "Pval"])
        for j in other_cols:
            model = regressor(y, X[selected_cols + [j]])
            pvals = pvals.append(pd.DataFrame([[j, model.pvalues[j]]], columns=["Cols", "Pval"]), ignore_index=True)
        pvals = pvals.sort_values(by=["Pval"]).reset_index(drop=True)
        pvals = pvals[pvals.Pval <= sl]
        if pvals.shape[0] > 0:

            model = regressor(y, X[selected_cols + [pvals["Cols"][0]]])
            iterations_log += str("\nEntered : " + pvals["Cols"][0] + "\n")
            iterations_log += "\n\n" + str(model.summary()) + "\nAIC: " + str(model.aic) + "\nBIC: " + str(
                model.bic) + "\n\n"

            if elimination_criteria == "aic":
                new_criteria = model.aic
                if new_criteria < criteria:
                    print("Entered :", pvals["Cols"][0], "\tAIC :", model.aic)
                    selected_cols.append(pvals["Cols"][0])
                    other_cols.remove(pvals["Cols"][0])
                    criteria = new_criteria
                else:
                    print("break : Criteria")
                    break
            elif elimination_criteria == "bic":
                new_criteria = model.bic
                if new_criteria < criteria:
                    print("Entered :", pvals["Cols"][0], "\tBIC :", model.bic)
                    selected_cols.append(pvals["Cols"][0])
                    other_cols.remove(pvals["Cols"][0])
                    criteria = new_criteria
                else:
                    print("break : Criteria")
                    break
            elif elimination_criteria == "r2" and model_type == "linear":
                new_criteria = model.rsquared
                if new_criteria > criteria:
                    print("Entered :", pvals["Cols"][0], "\tR2 :", model.rsquared)
                    selected_cols.append(pvals["Cols"][0])
                    other_cols.remove(pvals["Cols"][0])
                    criteria = new_criteria
                else:
                    print("break : Criteria")
                    break
            elif elimination_criteria == "adjr2" and model_type == "linear":
                new_criteria = model.rsquared_adj
                if new_criteria > criteria:
                    print("Entered :", pvals["Cols"][0], "\tAdjR2 :", model.rsquared_adj)
                    selected_cols.append(pvals["Cols"][0])
                    other_cols.remove(pvals["Cols"][0])
                    criteria = new_criteria
                else:
                    print("Break : Criteria")
                    break
            else:
                print("Entered :", pvals["Cols"][0])
                selected_cols.append(pvals["Cols"][0])
                other_cols.remove(pvals["Cols"][0])

        else:
            print("Break : Significance Level")
            break

    model = regressor(y, X[selected_cols])
    if elimination_criteria == "aic":
        criteria = model.aic
    elif elimination_criteria == "bic":
        criteria = model.bic
    elif elimination_criteria == "r2" and model_type == "linear":
        criteria = model.rsquared
    elif elimination_criteria == "adjr2" and model_type == "linear":
        criteria = model.rsquared_adj

    print(model.summary())
    print("AIC: " + str(model.aic))
    print("BIC: " + str(model.bic))
    print("Final Variables:", selected_cols)

    return selected_cols, iterations_log


def forwardSelection(X, y, model_type="linear", elimination_criteria="aic", varchar_process="dummy_dropfirst", sl=0.05):
    """
    Forward Selection is a function, based on regression models, that returns significant features and selection iterations.\n
    Required Libraries: pandas, numpy, statmodels

    Parameters
    ----------
    X : Independent variables (Pandas Dataframe)\n
    y : Dependent variable (Pandas Series, Pandas Dataframe)\n
    model_type : 'linear' or 'logistic'\n
    elimination_criteria : 'aic', 'bic', 'r2', 'adjr2' or None\n
        'aic' refers Akaike information criterion\n
        'bic' refers Bayesian information criterion\n
        'r2' refers R-squared (Only works on linear model type)\n
        'r2' refers Adjusted R-squared (Only works on linear model type)\n
    varchar_process : 'drop', 'dummy' or 'dummy_dropfirst'\n
        'drop' drops varchar features\n
        'dummy' creates dummies for all levels of all varchars\n
        'dummy_dropfirst' creates dummies for all levels of all varchars, and drops first levels\n
    sl : Significance Level (default: 0.05)\n


    Returns
    -------
    columns(list), iteration_logs(str)\n\n
    Not Returns a Model


    See Also
    --------
    https://en.wikipedia.org/wiki/Stepwise_regression
    """
    X = __varcharProcessing__(X, varchar_process=varchar_process)
    return __forwardSelectionRaw__(X, y, model_type=model_type, elimination_criteria=elimination_criteria, sl=sl)