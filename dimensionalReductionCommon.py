from common import *


class Dimensional_Reduction_Common:
    def percentaje_variance_detected(self, variance, index):
        percentaje = variance.loc[:, 0:index].sum(axis=1).values[0]
        return percentaje

    def trucante_components(self, components, variance):
        for component in range(components):
            percentaje = self.percentaje_variance_detected(variance, component + 1)
            if percentaje > 0.98 and component > 2:
                return component - 1

        return component

    def transform_data(self, transformer, data, trucanted_components):
        transformed_data = transformer.transform(data)
        columns = ["Comp_{}".format(index + 1) for index in range(trucanted_components)]
        transformed_data = pd.DataFrame(transformed_data, columns=columns)
        return transformed_data

    def error_SE(self, data, data_approx):
        distance = (np.array(data) - np.array(data_approx)) ** 2
        square_error_sum = np.sum(distance, axis=1)
        square_error_sum_df = pd.Series(data=square_error_sum, index=data.index)
        square_error_sum_df_scale = (
            square_error_sum_df - np.min(square_error_sum_df)
        ) / (np.max(square_error_sum_df) - np.min(square_error_sum_df))
        return square_error_sum_df_scale

    def get_error(self, data, transformed_data, pca):
        data_approx = pca.inverse_transform(transformed_data)
        square_error_sum_scale = self.error_SE(data, data_approx)
        return np.mean(square_error_sum_scale)

    def concat_vertical(self, df1, df2):
        return pd.concat([df1, df2]).reset_index().drop("index", axis=1)
