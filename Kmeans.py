from common import *


class KMEANS_:
    def add_labels_to_data(self, data, metadata):
        transformed_data = data.copy()
        transformed_data["labels_kmeans"] = metadata["labels"]
        return transformed_data

    def get_metrics(self, method, metadata, data, feature):
        labels = metadata["labels"]
        real = data[feature]
        metrics = [
            method,
            rand_score(labels, real),
            adjusted_rand_score(labels, real),
            v_measure_score(labels, real),
            fowlkes_mallows_score(labels, real),
            homogeneity_score(labels, real),
            normalized_mutual_info_score(labels, real),
        ]
        metrics_df = pd.DataFrame(
            [metrics],
            columns=[
                "method",
                "rand_score",
                "adjusted_rand_score",
                "v_measure_score",
                "fowlkes_mallows_score",
                "homogeneity_score",
                "normalized_mutual_info_score",
            ],
        )
        return metrics_df

    def apply(self, data, method="", classification=False, feature=None):
        clusters = 5
        error = 1e-3
        max_iter = 1000
        metadata = {}

        k_means = KMeans(
            n_clusters=clusters, max_iter=max_iter, tol=error, random_state=17
        )

        if classification:
            data_features = data.drop("finance", axis=1)
            k_means.fit(data_features)
            metadata["labels"] = k_means.labels_
            metadata["metrics"] = self.get_metrics(method, metadata, data, feature)
        else:
            k_means.fit(data)
            metadata["labels"] = k_means.labels_

        metadata["kmeans_transformer"] = k_means
        metadata["centroids"] = k_means.cluster_centers_
        metadata["data"] = self.add_labels_to_data(data, metadata)

        return metadata

    def plotKmeans(self, raw_data, metadata, feature1, feature2):
        labels = metadata["data"]["labels_kmeans"].unique()
        x_1 = raw_data[feature1]
        x_2 = raw_data[feature2]
        colors = ["black", "m", "g", "y", "r"]
        fig = plt.figure(figsize=(5, 5))

        for label in labels:
            plt.plot(
                x_1[metadata["labels"] == label],
                x_2[metadata["labels"] == label],
                "+",
                color=colors[label],
            )

        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
