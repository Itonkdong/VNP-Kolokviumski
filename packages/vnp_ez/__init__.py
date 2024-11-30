import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder


def missing_table(data: pd.DataFrame):
    statistics_missing_table = data.isnull().sum().reset_index().rename(columns={"index": "Feature", 0: "CountMissing"})
    percentage_missing = (
        (statistics_missing_table["CountMissing"] / len(data) * 100).reset_index()).reset_index().rename(
        columns={"CountMissing": "PercentageMissing"})
    statistics_missing_table["PercentageMissing"] = percentage_missing["PercentageMissing"]
    statistics_missing_table["Total"] = len(data)
    return statistics_missing_table


def balance_table(data: pd.DataFrame, target_column):
    balance_table = data.groupby(target_column).size().reset_index().rename(
        columns={target_column: "Class", 0: "Count"})
    tmp = (balance_table["Count"] / len(data) * 100).reset_index().rename(
        columns={target_column: "Class", "Count": "Percentage"})
    balance_table["Percentage"] = tmp["Percentage"]
    balance_table["Total"] = len(data)
    return balance_table


def get_numerical_features_names(data: pd.DataFrame):
    result_columns = []
    for column in data.columns:
        if pd.api.types.is_any_real_numeric_dtype(data[column]):
            result_columns.append(column)

    return result_columns


def get_categorical_features_names(data: pd.DataFrame, target_feature=None):
    categorical_features = []
    for column in data.columns:
        if target_feature is not None and column == target_feature:
            continue

        if not pd.api.types.is_any_real_numeric_dtype(data[column]):
            categorical_features.append(column)
    return categorical_features


def get_missing_features_name(data: pd.DataFrame):
    missing_values_features_names = []
    for column in data.columns:
        if len(data[data[column].isna()]) > 0:
            missing_values_features_names.append(column)
    return missing_values_features_names


def show_displots(data: pd.DataFrame, columns: list):
    for column in columns:
        sns.displot(data, x=column, kde=True)
        plt.show()


def show_displot_before_and_after_inputation(data_before: pd.DataFrame, data_after: pd.DataFrame, missing_feature_name):
    sns.displot(data_before[missing_feature_name], kde=True)
    plt.show()
    sns.displot(data_after[missing_feature_name], kde=True)
    plt.show()


def correlated_imputer(data: pd.DataFrame, correlated_columns: list, strategy="knn", n_neighbours=5):
    df_copy = data.copy()
    if strategy == "knn":
        knn_imputer = KNNImputer(n_neighbors=n_neighbours)
        normalizer = MinMaxScaler()
        normalized_values = normalizer.fit_transform(df_copy[correlated_columns])
        filled_with_knn = knn_imputer.fit_transform(normalized_values)
        imputed = normalizer.inverse_transform(filled_with_knn)
    elif strategy == "mice":
        mice = IterativeImputer()
        imputed = mice.fit_transform(df_copy[correlated_columns])

    df_copy[correlated_columns] = imputed

    return df_copy


def uncorrelated_imputer(data: pd.DataFrame, missing_feature: str, strategy="mean", const_value=-1):
    df_copy = data.copy()
    if strategy != "const":
        fill_value = getattr(df_copy[missing_feature], strategy)()
        if strategy == "mode":
            fill_value = fill_value[0]
        df_copy[missing_feature] = df_copy[missing_feature].fillna(fill_value)
    else:
        df_copy[missing_feature] = df_copy[missing_feature].fillna(const_value)
    return df_copy


def impute_data(data: pd.DataFrame, data_to_impute: list, strategies: list):
    if len(data_to_impute) != len(strategies):
        return "Differenet lengts"
    data_copy = data.copy()
    for impute_data, strategie in zip(data_to_impute, strategies):
        if strategie == "mice" or strategie == "knn":
            data_copy = correlated_imputer(data_copy, impute_data, strategie)
        else:
            if len(impute_data) != 1:
                return "Impute failed: Uncorrelated imputing with multiple data"
            data_copy = uncorrelated_imputer(data_copy, impute_data[0], strategie)
    return data_copy


def encode_data(data: pd.DataFrame, features_to_encode: list):
    encoders = {}
    data_copy = data.copy()
    for feature in features_to_encode:
        encoder = OrdinalEncoder()
        data_copy[[feature]] = encoder.fit_transform(data_copy[[feature]])
        encoders[feature] = encoder
    return data_copy, encoders


def show_ba_displots_pairwise(data_before: pd.DataFrame, data_after: pd.DataFrame, features: list):
    for feature in features:
        show_displot_before_and_after_inputation(data_before, data_after, missing_feature_name=feature)
