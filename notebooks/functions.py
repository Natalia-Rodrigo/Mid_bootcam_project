
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
    df2 = DropSamples(df).copy()

    sted_df = df2.groupby(tumor_column).std().reset_index()
    sted_df_melt = sted_df[sted_df[tumor_column] == tumor_value].drop(tumor_column, axis=1).melt()

    return sted_df_melt[sted_df_melt['value']<1]['variable'].to_list()


def getListDifferentialGenes(df:pd.DataFrame, psig=0.05, tumor_value = 'tumoral', tumor_column = 'type'):
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


def trainLogisticModel(df:pd.DataFrame, ColumnToPredict= 'type'):
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

    return LogisticModelView(model=classification, y_pred=y_pred, y_test=y_test)


def predict(model:LogisticRegression, X):
    return LogisticModelView(model=model, y_pred=model.predict(X=X), y_test=X)

def LogisticModelView(model:LogisticRegression, y_pred, y_test):

    # Create Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(cm,display_labels=model.classes_)
    disp.plot()

    plt.show()

    # Print Precision of CM
    scores = {
        'accuary' : accuracy_score(y_test,y_pred),
        'precision' : precision_score(y_test,y_pred),
        'recall' : recall_score(y_test,y_pred),
        'f1' : f1_score(y_test,y_pred),
        'kappa' : cohen_kappa_score(y_test,y_pred)
    }

    return model, cm, scores


def quantileTransformer(df:pd.DataFrame, qt = None):
    df_transformed = DropSamples(df).copy()

    if qt is None:
        # Fit new dataset, if none is given
        qt = QuantileTransformer (output_distribution ='normal')
        qt.fit(df_transformed[df_transformed.columns[2:]])
    
    # Apply Transformation to data set
    df_transformed[df_transformed.columns[2:]] = qt.transform(df_transformed[df_transformed.columns[2:]])
    return df_transformed, qt
    
