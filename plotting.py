import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import pandas as pd

def plot_dendrogram(df, title):
    """
    Plot a dendrogram showing the hierarchical clustering of the columns of the given dataframe.
    This function uses hierarchical clustering to group highly correlated features together and show the results in a
    dendrogram, which can be more readable than a correlation heatmap.

    Parameters
    ----------
    df : pandas DataFrame
        The input dataframe containing the columns to cluster.
    title : str
        The title of the plot.
    """
    corr_matrix = df.corr().abs()
    corr_condensed = hierarchy.distance.squareform(1 - corr_matrix)
    z = hierarchy.linkage(corr_condensed, method='ward')
    fig, ax = plt.subplots(figsize=(15,10))
    dendrogram = hierarchy.dendrogram(z, labels=corr_matrix.columns, ax=ax, leaf_rotation=90)
    plt.title(title)
    plt.show()



def highlight_min_max(df, max_color='yellow', min_color='green'):
    """
    Returns a copy of a DataFrame with background colors highlighting the minimum and maximum value in each column.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to be highlighted.

    color_min : str, optional (default='green')
        The color to highlight the minimum value in each column. Must be a valid CSS color string.

    color_max : str, optional (default='yellow')
        The color to highlight the maximum value in each column. Must be a valid CSS color string.

    Returns:
    --------
    pd.DataFrame
        A copy of the input DataFrame with background colors highlighting the minimum and maximum value in each column.
    """
    def highlight_min_max_func(s):
        is_min = s == s.min()
        is_max = s == s.max()
        return [f'background-color: {max_color}' if v else f'background-color: {min_color}' if w else '' 
                for v, w in zip(is_max, is_min)]

    return df.style.apply(highlight_min_max_func, axis=0)

def plot_correlation_heatmap(df, title, **kwargs):
    corr = df.corr()
    heatmap = sns.heatmap(corr, annot=True, fmt='.2f', **kwargs)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), ha='right', rotation=15)
    heatmap.set_title(title)
