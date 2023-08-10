### Regression

This is a sample project for regression using: 
- Simple Linear Regression (SLR) 
- Multiple Linear Regression (MLR)

SLRs models are developed on two different variables. 
* For MLR, the model uses the list *X_columns* in the file *params.py* as the predictor variables.
* For SLR_Var1, the model uses first variable in the list *X_columns* as the predictor variable.
* For SLR_Var2, the model uses second variable in the list *X_columns* as the predictor variable.

1. To install required packages for your environment from *requirements.txt*, run the following command:   
*$ pip install -r requirements.txt*  

2. To run the project from the jupyter notebook, 
    - First, uncomment (if commented) the following second line of code in *src/visualization.py* as below:  
      *get_ipython().run_line_magic('matplotlib', 'inline')*  
    
    - Second, execute the following code in the *code cell*:    
      *import regression*  

3. To run the regression project from the command line:  
    - First, comment (if uncommented) the second line of code in the file *src/visualization.py* as below:  
      *#get_ipython().run_line_magic('matplotlib', 'inline')*  
*$ python regression.py*

 4. Input parameters such as *predictor variable names* and *test_size* can be changed in the *params.py*.  

**Example output** of the project in jupyter notebook is available in the file ***src/regression.ipynb*.**  
**Data** output files are available in the ***output*** folder.  
**Plot** output files are available in the ***plots*** folder.  