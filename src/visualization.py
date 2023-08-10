"""Plot charts to visualize the model performance."""

from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_boxplot(list_df, titles, num_run):

    # Create 4 boxplots (MSE, MAE, R-Squared and PCC).
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,5))
    index = 0  
    
    for row in ax:
        for ax_cur in row:            
            list_df[index].plot(kind='box', ax=ax_cur)
            ax_cur.set_title(titles[index])
            index+=1
    plt.tight_layout()
    plt.subplots_adjust(hspace=.6, top=0.85)
    plt.suptitle(f' Performance of Simple and Multiple Linear Reg in {num_run} iterations', fontweight ="bold")
    plt.savefig(f'../plots/Metrics_{num_run}_box.png',bbox_inches='tight')
    plt.show()   

def plot_bar_charts(list_df_metrics, titles):

    # Create 4 bar charts (MSE, MAE, R-Squared and PCC).
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8,5))
    index = 0
    for row in ax:
        for ax_cur in row:
            ymin = list_df_metrics.iloc[:,index].values.min()
            ymax = list_df_metrics.iloc[:,index].values.max()

            list_df_metrics.iloc[:,index].plot(kind='bar', ax = ax_cur, color = 'b')  
            ax_cur.set_title(titles[index])      
            ax_cur.set_ylim(bottom=ymin*(1-0.05),top=ymax*(1+0.1))
            ax_cur.set_xticklabels(list_df_metrics.index, rotation = 45)
            ax_cur.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
               alpha=0.5)
            plt.xticks(rotation = 45)
            index+=1   
    plt.tight_layout()
    plt.subplots_adjust(hspace=.9, top=0.9)
    
    plt.suptitle('Average performance', fontweight ="bold")
    num_run = list_df_metrics.iloc[:,[0]].columns[0]
    plt.savefig(f'../plots/Metrics_{num_run}_bar.png',bbox_inches='tight')
    plt.show()     


