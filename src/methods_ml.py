from sklearn.linear_model import LinearRegression
import analysis

# Implementation of the machine learning methods
def ml_simple_linear_reg(X_train, X_test, y_train, y_test, X_indx = 0 ):
    lr = LinearRegression().fit(X_train[:, X_indx:X_indx+1],y_train)
    yhat_lr = lr.predict(X_test[:, X_indx:X_indx+1])
    metrics = analysis.evaluate_models(y_test, yhat_lr)
    return metrics

def ml_multiple_linear_reg(X_train, X_test, y_train, y_test):
    lr = LinearRegression().fit(X_train,y_train)
    yhat_lr = lr.predict(X_test)
    metrics = analysis.evaluate_models(y_test, yhat_lr)
    return metrics
   
