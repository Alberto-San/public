from common import *


class CategoricalPreprocessing:
    def count_null_categorical(self, df, columns_categorical):
        rows = []
        for column in columns_categorical:
            df_categorical = df[column].replace(np.nan, "NaN")
            rows.append([column, df_categorical[df_categorical == "NaN"].count()])
        return pd.DataFrame(rows, columns=["Column", "Amount"])

    def getIndexesOfNaN(self, data_transformed):
        indexes = {}
        for column in data_transformed.columns:
            null_values = pd.isnull(data_transformed)
            indexes[column] = null_values.index[null_values[column] == True]
        return indexes

    def getEncodedWithNaNData(self, data_transformed, categorical_columns, indexes):
        LE = defaultdict(LabelEncoder)
        data_categorical_transformed = data_transformed[categorical_columns]
        data_encoded = data_categorical_transformed.apply(
            lambda x: LE[x.name].fit_transform(x)
        )
        data_processed = data_encoded.copy()
        columns = data_transformed.columns

        for column in columns:
            if len(indexes[column]) > 0:
                data_processed.loc[indexes[column], column] = np.NaN

        return data_processed

    def convert_categorical_to_number(self, df, columns):
        df_categorical = df[columns]
        indexes_nan = self.getIndexesOfNaN(df_categorical)
        data_processed = self.getEncodedWithNaNData(
            df_categorical, columns, indexes_nan
        )
        df_final = df.copy()
        for column in columns:
            df_final[column] = data_processed[column]
        return df_final
