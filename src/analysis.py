#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def plot_boxplot(list_df, titles, num_run):
    # Create 4 boxplots (MSE, MAE, R-Squared and PCC) in the same file.
    fig, ax = plt.subplots(nrows=2, ncols=2)
    index = 0    
    for row in ax:
        for ax_cur in row:            
            list_df[index].plot(kind='box', ax=ax_cur)
            ax_cur.set_title(titles[index])
            index+=1
    plt.tight_layout()
    plt.subplots_adjust(hspace=.6, top=0.9)
    plt.suptitle(f' Performance of Simple and Multiple Linear Reg in {num_run} iterations', fontweight ="bold")
    plt.savefig(f'../plots/Metrics_{num_run}_box.png',bbox_inches='tight')
    plt.show()   

def plot_bar_charts(list_df_metrics, titles):
    # Create 4 bar charts (MSE, MAE, R-Squared and PCC) in the same file.
    # Each bar chart compares the random_split results against cross-validation results.
    fig, ax = plt.subplots(nrows=2, ncols=2)
    index = 0
    for row in ax:
        for ax_cur in row:
            ymin = list_df_metrics.iloc[:,index].values.min()
            ymax = list_df_metrics.iloc[:,index].values.max()

            list_df_metrics.iloc[:,index].plot(kind='bar', ax = ax_cur)  
            ax_cur.set_title(titles[index])      
            ax_cur.set_ylim(bottom=ymin*(1-0.05),top=ymax*(1+0.1))
            ax_cur.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
            index+=1   
    plt.tight_layout()
    plt.subplots_adjust(hspace=.9, top=0.9)
    plt.suptitle('Average performance', fontweight ="bold")
    num_run = list_df_metrics.iloc[:,[0]].columns[0]
    plt.savefig(f'../plots/Metrics_{num_run}_bar.png',bbox_inches='tight')
    plt.show()     

def evaluate_models(y_test, yhat):
    # Calculate Mean Squared Error (MSE), Mean Absolute Error (MAE), 
    # r-squared (R2), and Pearson's Correlation Coef(PCC)
    metrics = dict()
    metrics['mse'] = mean_squared_error(y_test, yhat)
    metrics['mae'] = mean_absolute_error(y_test, yhat)
    metrics['r2'] = r2_score(y_test, yhat)
    metrics['corr'], _ = pearsonr(y_test, yhat)
    return metrics
