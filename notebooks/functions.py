
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score

def DropSamples(df:pd.DataFrame):
    if 'samples' in df.columns:
        df = df.drop('samples', axis=1)
    return df


def calculate_difference(df:pd.DataFrame, tumor_column = 'type'):
    """
    Calculate the mean difference between two groups in a DataFrame based on the specified 'tumor_column'.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing data for comparison.
    - tumor_column (str, optional): The column defining the groups. Default is 'type'.

    Returns:
    - pd.DataFrame: A new DataFrame containing the means of each column for each group, 
                    as well as the differences between the means of the two groups.
    """    
    df2 = DropSamples(df).copy()

    means_df = df2.groupby(tumor_column).agg('mean').reset_index()

    # define Min_values Dict
    min_values = {}
    for col in means_df.columns[1:]:
        min_values[col] = means_df[col][0] - means_df[col][1]

    min_values_df = pd.DataFrame([min_values])
    min_values_df[tumor_column] = 'difference'

    # return new concatted dataframe
    return pd.concat([means_df, min_values_df], axis=0)


def plotSTDEV(df:pd.DataFrame, tumor_value = 'tumoral', tumor_column = 'type'):
    """
    Plot the standard deviation (STDEV) distribution of a specified 'tumor_value' within a DataFrame,
    grouped by the 'tumor_column'. The function generates a boxplot and a histogram for visualizing the distribution.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing data for visualization.
    - tumor_value (str, optional): The specific value within the 'tumor_column' for which the STDEV is plotted.
                                  Default is 'tumoral'.
    """
    df2 = DropSamples(df).copy()

    sted_df = df2.groupby(tumor_column).std().reset_index()
    sted_df_melt = sted_df[sted_df[tumor_column] == tumor_value].drop(tumor_column, axis=1).melt()

    f, ax  = plt.subplots(2,1,figsize=(10,5), sharex=True)
    ax[0] = sns.boxplot(sted_df_melt, ax=ax[0], orient='h')
    ax[1] = sns.histplot(sted_df_melt, ax=ax[1])

    plt.show()

    return sted_df_melt


# make function to plot the thing that we talked about

def getLowSTEDVList(df:pd.DataFrame, tumor_value = 'tumoral', tumor_column = 'type'):
    """
    Get a list of variables with a standard deviation (STDEV) less than 1 for a specific tumor group in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing data for analysis.
    - tumor_value (str, optional): The specific tumor group to analyze. Default is 'tumoral'.
    - tumor_column (str, optional): The column defining the tumor groups. Default is 'type'.

    Returns:
    - list: A list of variables with STDEV values less than 1 for the specified tumor group.
    """
    df2 = DropSamples(df).copy()

    sted_df = df2.groupby(tumor_column).std().reset_index()
    sted_df_melt = sted_df[sted_df[tumor_column] == tumor_value].drop(tumor_column, axis=1).melt()

    return sted_df_melt[sted_df_melt['value']<1]['variable'].to_list()


def getListDifferentialGenes(df:pd.DataFrame, psig=0.05, tumor_value = 'tumoral', tumor_column = 'type'):
    """
    Identify differentially expressed genes between a specified tumor group and the rest of the samples in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing gene expression data.
    - psig (float, optional): Significance threshold for identifying differentially expressed genes. Default is 0.05.
    - tumor_value (str, optional): The specific tumor group for which differential expression is assessed. Default is 'tumoral'.
    - tumor_column (str, optional): The column defining the tumor groups. Default is 'type'.

    Returns:
    - list: A list of gene names that are differentially expressed in the specified tumor group based on a t-test.
    """    
    df2 = DropSamples(df).copy()

    df_cancer = df2[df2[tumor_column]==tumor_value]
    df_normal = df2[df2[tumor_column]!=tumor_value]

    differential_expressed_genes = []
    for col in df_cancer.columns[2:]:
        t, pvalue= st.ttest_ind(df_cancer[col],df_normal[col], equal_var = False, alternative = 'two-sided')
        if pvalue < psig:
            differential_expressed_genes.append(str(col))

    return differential_expressed_genes


def find_correlated_genes(df:pd.DataFrame, threshold=0.95):
    """
    Identify and analyze genes with high correlation coefficients in a gene expression DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing gene expression data.
    - threshold (float, optional): The correlation coefficient threshold for identifying highly correlated genes. Default is 0.95.

    Returns:
    - Tuple: A tuple containing three elements:
        1. List: Information about correlated genes, including column names, correlation values, and index names.
        2. List: Names of genes to be excluded to avoid redundancy.
        3. pd.DataFrame: The correlation matrix of the input DataFrame.
    """
    df2 = DropSamples(df).copy()
    
    r_values = df2.corr()
    
    correlated_genes_list = []
    # Create a mask for values above the threshold
    mask = (r_values.to_numpy() > threshold) & (r_values.index.to_numpy() != r_values.columns.to_numpy()[:, None])

    # Extract the column and index names where the mask is True
    correlated_columns, correlated_rows = np.where(mask)

    for col, index in zip(r_values.columns[correlated_columns], r_values.index[correlated_rows]):
        value = r_values.at[index, col]
        correlated_genes_list.append([col, value, index])

    exclude_list = []
    included_columns = []

    for col in correlated_genes_list:
        if col not in included_columns:
            exclude_list.append(col[0])
            included_columns.append(col[2])

    return correlated_genes_list, exclude_list, r_values

def trainLogisticModel(df: pd.DataFrame, ColumnToPredict='type', tumor_value = 'tumoral'):
    """
    Train a logistic regression model to predict a specified column in a DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing data for training and testing the model.
    - ColumnToPredict (str, optional): The column to be predicted using logistic regression. Default is 'type'.
    - tumor_value (str, optional): The specific tumor group to be treated as the positive label. Default is 'tumoral'.

    Returns:
    - Tuple: A tuple containing three elements:
        1. LogisticRegression: Trained logistic regression model.
        2. pd.DataFrame: Confusion matrix for model evaluation.
        3. dict: Dictionary containing various classification scores (e.g., accuracy, precision, recall).
    """
    df2 = DropSamples(df).copy()

    # Split into X/Y based on ColumnToPredict
    y = df2[ColumnToPredict]
    X = df2.drop([ColumnToPredict], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1337)

    # Peform underSampler if values are not equal
    if df2[ColumnToPredict].value_counts()[0] != df2[ColumnToPredict].value_counts()[1]:
        rus = RandomUnderSampler(random_state=69)
        X_train, y_train = rus.fit_resample(X_train, y_train)

    # Create LogisticRegression model
    classification = LogisticRegression(random_state=0, solver='lbfgs')
    classification.fit(X_train, y_train)

    # Create Prediction values
    y_pred = classification.predict(X=X_test)

    # return y_pred, y_test
    model, cm, scores = LogisticModelView(model=classification, y_pred=y_pred, y_test=y_test.values, pos_label=tumor_value)
    return model, cm, scores

def predict(model: LogisticRegression, X, tumor_value = 'tumoral'):
    """
    Do not use, not ready yet. Maybe will never be ready tbh might not really need this one :)
    """
    model, cm, scores = LogisticModelView(model=model, y_pred=model.predict(X=X), y_test=X, pos_label=tumor_value)
    return model, cm, scores

def LogisticModelView(model: LogisticRegression, y_pred, y_test, pos_label):
    """
    Evaluate and visualize the performance of a logistic regression model using a confusion matrix.

    Parameters:
    - model (LogisticRegression): Trained logistic regression model.
    - y_pred: Predicted values from the model.
    - y_test: True labels for evaluation.
    - pos_label (str): The positive label, typically the specific tumor group.

    Returns:
    - Tuple: A tuple containing three elements:
        1. LogisticRegression: Trained logistic regression model.
        2. pd.DataFrame: Confusion matrix for model evaluation.
        3. dict: Dictionary containing various classification scores (e.g., accuracy, precision, recall).
    """
    # Create Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    disp.plot()

    # Print Precision of CM
    scores = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_true=y_test, y_pred=y_pred, pos_label=pos_label),
        'recall': recall_score(y_test, y_pred, pos_label=pos_label),
        'f1': f1_score(y_test, y_pred, pos_label=pos_label),
        'kappa': cohen_kappa_score(y_test, y_pred)
    }

    return model, cm, scores



def quantileTransformer(df:pd.DataFrame, qt = None):
    """
    Apply quantile transformation to a DataFrame, transforming numerical columns to follow a normal distribution.

    Parameters:
    - df (pd.DataFrame): Input DataFrame containing numerical data for transformation.
    - qt (QuantileTransformer or None, optional): An existing QuantileTransformer object to apply, or None to fit a new one. Default is None.

    Returns:
    - Tuple: A tuple containing two elements:
        1. pd.DataFrame: Transformed DataFrame with numerical columns following a normal distribution.
        2. QuantileTransformer: The fitted or provided QuantileTransformer object.
    """
    df_transformed = DropSamples(df).copy()

    if qt is None:
        # Fit new dataset, if none is given
        qt = QuantileTransformer (output_distribution ='normal')
        qt.fit(df_transformed[df_transformed.columns[2:]])
    
    # Apply Transformation to data set
    df_transformed[df_transformed.columns[2:]] = qt.transform(df_transformed[df_transformed.columns[2:]])
    return df_transformed, qt
    
