import statsmodels.api as sm

from utils import __varcharProcessing__


def __backwardSelectionRaw__(X, y, model_type="linear", elimination_criteria="aic", sl=0.05):
    iterations_log = ""
    last_eleminated = ""
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

    for i in range(X.shape[1]):
        if i != 0:
            if elimination_criteria == "aic":
                criteria = model.aic
                new_model = regressor(y, X)
                new_criteria = new_model.aic
                if criteria < new_criteria:
                    print("Regained : ", last_eleminated)
                    iterations_log += "\n" + str(new_model.summary()) + "\nAIC: " + str(
                        new_model.aic) + "\nBIC: " + str(new_model.bic) + "\n"
                    iterations_log += str("\n\nRegained : " + last_eleminated + "\n\n")
                    break
            elif elimination_criteria == "bic":
                criteria = model.bic
                new_model = regressor(y, X)
                new_criteria = new_model.bic
                if criteria < new_criteria:
                    print("Regained : ", last_eleminated)
                    iterations_log += "\n" + str(new_model.summary()) + "\nAIC: " + str(
                        new_model.aic) + "\nBIC: " + str(new_model.bic) + "\n"
                    iterations_log += str("\n\nRegained : " + last_eleminated + "\n\n")
                    break
            elif elimination_criteria == "adjr2" and model_type == "linear":
                criteria = model.rsquared_adj
                new_model = regressor(y, X)
                new_criteria = new_model.rsquared_adj
                if criteria > new_criteria:
                    print("Regained : ", last_eleminated)
                    iterations_log += "\n" + str(new_model.summary()) + "\nAIC: " + str(
                        new_model.aic) + "\nBIC: " + str(new_model.bic) + "\n"
                    iterations_log += str("\n\nRegained : " + last_eleminated + "\n\n")
                    break
            elif elimination_criteria == "r2" and model_type == "linear":
                criteria = model.rsquared
                new_model = regressor(y, X)
                new_criteria = new_model.rsquared
                if criteria > new_criteria:
                    print("Regained : ", last_eleminated)
                    iterations_log += "\n" + str(new_model.summary()) + "\nAIC: " + str(
                        new_model.aic) + "\nBIC: " + str(new_model.bic) + "\n"
                    iterations_log += str("\n\nRegained : " + last_eleminated + "\n\n")
                    break
            else:
                new_model = regressor(y, X)
            model = new_model
            iterations_log += "\n" + str(model.summary()) + "\nAIC: " + str(model.aic) + "\nBIC: " + str(
                model.bic) + "\n"
        else:
            model = regressor(y, X)
            iterations_log += "\n" + str(model.summary()) + "\nAIC: " + str(model.aic) + "\nBIC: " + str(
                model.bic) + "\n"
        maxPval = max(model.pvalues)
        cols = X.columns.tolist()
        if maxPval > sl:
            for j in cols:
                if (model.pvalues[j] == maxPval):
                    print("Eliminated :", j)
                    iterations_log += str("\n\nEliminated : " + j + "\n\n")

                    del X[j]
                    last_eleminated = j
        else:
            break
    print(str(model.summary()) + "\nAIC: " + str(model.aic) + "\nBIC: " + str(model.bic))
    print("Final Variables:", cols)
    iterations_log += "\n" + str(model.summary()) + "\nAIC: " + str(model.aic) + "\nBIC: " + str(model.bic) + "\n"
    return cols, iterations_log


def backwardSelection(X, y, model_type="linear", elimination_criteria="aic", varchar_process="dummy_dropfirst",
                      sl=0.05):
    """
    Backward Selection is a function, based on regression models, that returns significant features and selection iterations.\n
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
    return __backwardSelectionRaw__(X, y, model_type=model_type, elimination_criteria=elimination_criteria, sl=sl)