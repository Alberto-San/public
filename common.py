import numpy as np
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor
from sklearn import preprocessing
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA, FastICA, KernelPCA
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd


def print_null_data(df):
    serie = df.isna().sum()
    df1 = pd.DataFrame(serie, columns=["amount"])
    df2 = df1[df1["amount"] > 0].sort_values("amount", ascending=False)
    return df2


def scaling_data(data, scaler):
    if scaler != "":
        if scaler == "robust":
            scaler = preprocessing.RobustScaler()
        elif scaler == "StandardScaler":
            scaler = preprocessing.StandardScaler()
        else:
            scaler = preprocessing.MinMaxScaler()
        df = scaler.fit_transform(data)
        transformation = scaler
    else:
        df = data
        transformation = None
    df = pd.DataFrame(df, columns=data.columns)
    return df, transformation


def get_categories_from_data(data, number_categories, labels):
    df = pd.cut(data, number_categories, labels=labels).cat.codes
    return df


def generate_categories(df, feature_continuous, number_categories):
    data_imputed_bins = df.copy()
    data = data_imputed_bins[feature_continuous]
    numeric_labels = [index for index in range(number_categories)]
    data_imputed_bins[feature_continuous] = get_categories_from_data(
        data, number_categories, numeric_labels
    )
    return data_imputed_bins


def get_scale_data_regression(data_imputed, data_dic_collection, scaler, feature):
    data_filtered, transformation = data_dic_collection[scaler]
    if scaler == "":
        df_transformed = data_filtered
    else:
        df_transformed = pd.DataFrame(
            transformation.transform(data_filtered), columns=data_imputed.columns
        )
        df_transformed[feature] = data_filtered[feature]
    return df_transformed


def concat_vertical(df1, df2):
    return pd.concat([df1, df2]).reset_index().drop("index", axis=1)


# def get_summary_algorithms_dimensional_reduction(
#     data_dic,
#     scaler,
#     flag_first=True,
#     summary_dim_reduction=None,
#     regression=False,
#     output=None,
# ):
#     algoritms = [PCA_(), KernelPCA_(), ICA_()]

#     for algorimth in algoritms:
#         data = data_dic[scaler]
#         if regression:
#             X, Y = data.drop(output, axis=1), data[output]
#             algorimth.apply(X, scaler)
#         else:
#             algorimth.apply(data, scaler)

#         if flag_first:
#             summary_dim_reduction = algorimth.metadata["summary"]
#             flag_first = False
#         else:
#             df = algorimth.metadata["summary"]
#             summary_dim_reduction = concat_vertical(summary_dim_reduction, df)

#     return summary_dim_reduction, flag_first

# def get_summary_scaler_dim_reduction(data_dic, regression=False, output=None):
# 	'''
# 	data_dic.keys() => names of the scaler
# 	data_dic[key] => dataframe
# 	regression: if regression, output will be drop from dataframe to be dim reduction, and output will stayed as originally.
# 	'''
# 	scalers = list(data_dic.keys())
# 	flag_first = True
# 	summary_dim_reduction = None

# 	for index in range(len(scalers)):
# 		scaler = scalers[index]
# 		summary_dim_reduction, flag_first = get_summary_algorithms_dimensional_reduction(
# 			data_dic,
# 			scaler,
# 			flag_first,
# 			summary_dim_reduction,
# 			regression=regression,
# 			output=output,
# 		)

# 	columns = [
# 		"Tipo de Escalador",
# 		"Metodo",
# 		"No Componentes (98% Varianza)",
# 		"Error Cuadratico Medio Escalado",
# 	]
# 	summary_dim_reduction = summary_dim_reduction[columns]
# 	summary_dim_reduction["Numero de Componentes Optimo"] = summary_dim_reduction[
# 		"No Componentes (98% Varianza)"
# 	]
# 	summary_dim_reduction = summary_dim_reduction.drop(
# 		"No Componentes (98% Varianza)", axis=1
# 	)
# 	return summary_dim_reduction

# def basic_kmeans_train(data_dic, classification=False, feature=None):
	# scalers = list(data_dic.keys())
	# algoritms_name = ["pca", "kernel pca", "ica"]
	# algoritms = [PCA_(), KernelPCA_(), ICA_()]
	# schema = [
	# 	"method",
	# 	"rand_score",
	# 	"adjusted_rand_score",
	# 	"v_measure_score",
	# 	"fowlkes_mallows_score",
	# 	"homogeneity_score",
	# 	"normalized_mutual_info_score",
	# ]
	# information = pd.DataFrame([], columns=schema)
	# metadata = {}
	# for scaler in scalers:
	# 	index = 0
	# 	for algorithm in algoritms:
	# 		title = scaler + " " + algoritms_name[index]
	# 		data_scaler = data_dic[scaler]
	# 		if classification:
	# 			algorithm.apply(data_scaler.drop(feature, axis=1), scaler)
	# 		else:
	# 			algorithm.apply(data_scaler, scaler)
	# 		transformed_data = algorithm.metadata["transformed_data"]
	# 		if classification:
	# 			transformed_data = pd.concat(
	# 				[transformed_data, data_scaler[feature]], axis=1
	# 			)
	# 			metadata_kmeans = getKMeans(
	# 				transformed_data,
	# 				method=title,
	# 				classification=classification,
	# 				feature=feature,
	# 			)
	# 			information = pd.concat([information, metadata_kmeans["metrics"]])
	# 		else:
	# 			metadata_kmeans = getKMeans(transformed_data, method=title)
	# 		index += 1
	# 		metadata[title] = metadata_kmeans

	# return metadata, information

# def process_outliers_labeled_custom(self, data, output_feature):
        # scalers = ["", "robust", "StandardScaler", "mixMax"]
        # summary_collection = []
        # data_dic_collection = {}
        # labels = list(data[output_feature].unique())

        # for scaler in scalers:
        #     scaled_data, transformation = scaling_data(data, scaler=scaler)
        #     summary_collection_local = pd.DataFrame(
        #         [],
        #         columns=["label", "percentaje_rejected", "number_rejected", "scaler"],
        #     )
        #     data_filtered_local = pd.DataFrame([], columns=data.columns)
        #     for label in labels:
        #         data_filtered, metadata = self.LOF(
        #             data,
        #             scaled_data,
        #             scaler,
        #             is_classification=True,
        #             label_value=label,
        #             label_feature=output_feature,
        #         )
        #         data_filtered_local = pd.concat([data_filtered_local, data_filtered])
        #         summary_collection_local = pd.concat(
        #             [summary_collection_local, metadata["outliers"]]
        #         )
        #     data_filtered_local = data_filtered_local.reset_index().drop(
        #         "index", axis=1
        #     )
        #     summary_collection.append(summary_collection_local)
        #     data_dic_collection[scaler] = [data_filtered_local, transformation]

        # summary_df = pd.concat(summary_collection)
        # return summary_df, data_dic_collection