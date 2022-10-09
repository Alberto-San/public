from common import *


class Remove_Outliers_LOF:
    def getLofMetadata(self, LOF, data, processed_data):
        NOF = LOF.negative_outlier_factor_
        ground_truth = np.ones(len(data), dtype=int)
        return {
            "radio_outiler": (NOF.max() - NOF) / (NOF.max() - NOF.min()),
            "ground_truth": np.ones(len(data), dtype=int),
            "n_errors": (processed_data != ground_truth).sum(),
        }

    def filterOutliersLOF(self, data, process_LOF_data, ground_truth):
        pos = np.where(process_LOF_data == ground_truth)
        pos = np.asarray(pos)
        indexes = np.hstack(pos)
        return indexes

    def processDataWithLOF(self, scaled_data, k=5):
        transformer = LocalOutlierFactor(
            n_neighbors=k, algorithm="auto", contamination=0.05, metric="euclidean"
        )
        processed_data = transformer.fit_predict(scaled_data)
        metadata = self.getLofMetadata(transformer, scaled_data, processed_data)
        index_data_filtered = self.filterOutliersLOF(
            data=scaled_data,
            process_LOF_data=processed_data,
            ground_truth=metadata["ground_truth"],
        )
        return index_data_filtered

    def get_LOF_with_labeled_data(
        self, data_or, scaled_data, scaler, label_value, label_feature
    ):
        filt = data_or[label_feature] == label_value
        scaled_data_filtered = scaled_data[filt].drop(label_feature, axis=1)
        data = data_or[filt]
        index_data_filtered = self.processDataWithLOF(scaled_data_filtered)
        filtered_data = data.iloc[index_data_filtered]
        columns = ["label", "percentaje_rejected", "number_rejected", "scaler"]
        percentaje_rejected = 1 - filtered_data.shape[0] / data.shape[0]
        number_rejected = data.shape[0] - filtered_data.shape[0]
        row = [label_value, percentaje_rejected, number_rejected, scaler]
        metadata = {"outliers": pd.DataFrame([row], columns=columns)}
        filtered_data[label_feature] = label_value
        return filtered_data, metadata

    def get_LOF_with_non_labeled_data(self, data_or, scaled_data, scaler):
        scaled_data_filtered = scaled_data
        data = data_or
        index_data_filtered = self.processDataWithLOF(scaled_data_filtered)
        filtered_data = data.iloc[index_data_filtered]
        columns = ["percentaje_rejected", "number_rejected", "scaler"]
        percentaje_rejected = 1 - filtered_data.shape[0] / data.shape[0]
        number_rejected = data.shape[0] - filtered_data.shape[0]
        row = [percentaje_rejected, number_rejected, scaler]
        metadata = {"outliers": pd.DataFrame([row], columns=columns)}
        return filtered_data, metadata

    def LOF(
        self,
        data_or,
        scaled_data,
        scaler,
        is_classification=False,
        label_value="",
        label_feature="",
    ):

        if is_classification:
            filtered_data, metadata = self.get_LOF_with_labeled_data(
                data_or, scaled_data, scaler, label_value, label_feature
            )
        else:
            filtered_data, metadata = self.get_LOF_with_non_labeled_data(
                data_or, scaled_data, scaler
            )

        filtered_data = filtered_data.reset_index().drop("index", axis=1)
        return filtered_data, metadata

    def process_outliers_non_labeled(self, data, regression=False, feature=None):
        scalers = ["", "robust", "StandardScaler", "mixMax"]
        summary_collection = []
        data_dic_collection = {}

        for scaler in scalers:
            scaled_data, transformation = scaling_data(data, scaler=scaler)

            if regression:
                data_filtered, metadata = self.LOF(
                    data, scaled_data.drop(feature, axis=1), scaler
                )
            else:
                data_filtered, metadata = self.LOF(data, scaled_data, scaler)

            summary_collection.append(metadata["outliers"])
            data_dic_collection[scaler] = [data_filtered, transformation]

        summary_df = pd.concat(summary_collection)
        return summary_df, data_dic_collection

    def plotAgainstOriginal(
        self, data, data_without_outliers, feature_1="tc", feature_2="cres"
    ):
        plt.scatter(data[feature_1], data[feature_2], color="r")
        plt.scatter(
            data_without_outliers[feature_1],
            data_without_outliers[feature_2],
            color="b",
        )
        plt.xlabel(feature_1)
        plt.ylabel(feature_2)

    def plot_outliers(self, data, data_dic_collection):
        plt.figure(figsize=(28, 10))
        scalers = list(data_dic_collection.keys())
        for index in range(len(scalers)):
            scaler = scalers[index]
            data_filtered, transformation = data_dic_collection[scaler]
            plt.subplot(1, len(scalers), index + 1)
            plotAgainstOriginal(data, data_filtered)
            plt.title(scaler)
        plt.show()
