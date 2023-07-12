This is a sample project for regression using: 
- Simple Linear Regression (SLR) 
- Multiple Linear Regression (MLR)

SLRs models are developed on two different variables. 
# For MLR, the model uses the list *X_columns* in the file *params.py* as the predictor variables.
# For SLR_Var1, the model uses first variable in the list *X_columns* as the predictor variable.
# For SLR_Var2, the model uses second variable in the list *X_columns* as the predictor variable.

1. To install required packages for your environment from *requirements.txt*, run the following command:   
*$ pip install -r requirements.txt*  

2. Input parameters such as *test_size* can be changed in the *params.py*.

3. To run the regression project from the *src* directory:   
*$ python regression.py*

4. To run the project from the jupyter notebook, execute following command in the *code cell*:   
*import regression*  

Outputs of the project in jupyter notebook is available in the *regression.ipynb*.   
While running in the notebook, uncomment the first two lines in *analysis.py* for importing IPython, if necessary.

