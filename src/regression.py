import numpy as np
import pandas as pd
import methods_ml, params
import preprocess, analysis
import warnings
warnings.filterwarnings("ignore")

model_names = ['SLR_Var1', 'SLR_Var2', 'MLR']
titles = ['MSE', 'MAE', 'R-Squared', 'PCC']
np.random.seed(params.seed)

def random_split():  
    # Perform random-split to check the out-of-sample predictive performance
    # Perform regression for num_run no. of iterations
    # Select random test set in each iteration

    df_mse = pd.DataFrame(columns=model_names)
    df_mae = pd.DataFrame(columns=model_names)
    df_r2 = pd.DataFrame(columns=model_names)
    df_corr = pd.DataFrame(columns=model_names)
    
    df_list = [df_mse, df_mae, df_r2, df_corr]    
    df_avg = pd.DataFrame() # empty dataframe to save the metrics averages
    filenames = ['MSE', 'MAE', 'R-Squared', 'PCC']

    for i in range(params.num_run):
        # data i.e. X_train, X_test, y_train, y_test are selected in each iteration
        data = preprocess.split_data(X, y, params.test_size)
        df_list = iteration(i, data, *df_list)

    for i in range(len(df_list)):
        df_list[i].to_csv(f"../output/{filenames[i]}{params.num_run}.csv", index=False)

        # Save the averages of the metrics into the dataframes.         
        df_avg[f'{filenames[i]}_Avg{params.num_run}'] = df_list[i].mean(axis=0).to_frame()
          
    # Create boxplots for mean squared error (MSE), mean absolute error (MAE), 
    # r-squared (R-2) and Pearson's correlation coefficient (PCC) 
    # for all iterations to check for variance, outliers, skewness.
    print(f'Mean Squared Error (MSE), Mean Abs Error(MAE), R-Squared (R-2) and Pearson\'s Correlation Coef (PCC) of '
          f'\n1. Simple Linear Regression for the first variable (SLR_var1),'
          f'\n2. Simple Linear Regression for the second variable (SLR_var2), and '
          f'\n3. Multiple Linear Regression (MLR)')
    analysis.plot_boxplot(df_list, titles, params.num_run)  
    print("\n\nAverage model performance:") 
    print(df_avg)
    analysis.plot_bar_charts(df_avg, titles)   

def iteration(iterno, data, df_mse, df_mae, df_r2, df_corr):    
    # In each iteration, values are predicted using 3 different inputs
    # 1. Simple Linear Regression for first variable in data
    # 2. Simple Linear Regression for second variable in data
    # 3. Multiple Linear Regression for all the variables in data
    metrics_slr1 = methods_ml.ml_simple_linear_reg(*data, 0)
    metrics_slr2 = methods_ml.ml_simple_linear_reg(*data, 1)
    metrics_mlr = methods_ml.ml_multiple_linear_reg(*data)       
    df_mse_new = pd.DataFrame([{'SLR_Var1': metrics_slr1['mse'], 'SLR_Var2': metrics_slr2['mse'], 
                               'MLR': metrics_mlr['mse']}])
    df_mae_new = pd.DataFrame([{'SLR_Var1': metrics_slr1['mae'], 'SLR_Var2': metrics_slr2['mae'], 
                               'MLR': metrics_mlr['mae']}])
    df_r2_new = pd.DataFrame([{'SLR_Var1': metrics_slr1['r2'], 'SLR_Var2': metrics_slr2['r2'], 
                               'MLR': metrics_mlr['r2']}])
    df_corr_new = pd.DataFrame([{'SLR_Var1': metrics_slr1['corr'], 'SLR_Var2': metrics_slr2['corr'], 
                               'MLR': metrics_mlr['corr']}])
    # add new row for this iteration to the dataframes
    df_mse = pd.concat([df_mse, df_mse_new]) 
    df_mae = pd.concat([df_mae, df_mae_new]) 
    df_r2 = pd.concat([df_r2, df_r2_new]) 
    df_corr = pd.concat([df_corr, df_corr_new]) 
    return [df_mse, df_mae, df_r2, df_corr]

X, y = preprocess.get_data()
X = preprocess.normalize_data(X, X)
random_split()
