"""Implement machine learning(ML) models.

Return the Mean Squared Error (MSE), Mean Absolute Error (MAE), 
r-squared (R2), and Pearson's Correlation Coef(PCC) of each model.
"""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def ml_simple_linear_reg(X_train, X_test, y_train, y_test, X_indx = 0 ):
    lr = LinearRegression().fit(X_train[:, X_indx:X_indx+1],y_train)
    yhat_lr = lr.predict(X_test[:, X_indx:X_indx+1])
    metrics = evaluate_models(y_test, yhat_lr)
    return metrics

def ml_multiple_linear_reg(X_train, X_test, y_train, y_test):
    lr = LinearRegression().fit(X_train,y_train)
    yhat_lr = lr.predict(X_test)
    metrics = evaluate_models(y_test, yhat_lr)
    return metrics
   
def evaluate_models(y_test, yhat):

    # Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), 
    # r-squared (R2), and Pearson's Correlation Coef(PCC)
    metrics = dict()
    metrics['mse'] = mean_squared_error(y_test, yhat)
    metrics['mae'] = mean_absolute_error(y_test, yhat)
    metrics['r2'] = r2_score(y_test, yhat)
    metrics['corr'], _ = pearsonr(y_test, yhat)
    return metrics