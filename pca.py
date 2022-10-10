from public.common import *
from public.dimensionalReductionCommon import Dimensional_Reduction_Common


class PCA_(Dimensional_Reduction_Common):
    def __init__(self):
        self.metadata = {}
        self.schema = [
            "Metodo",
            "Tipo de Escalador",
            "No Componentes (98% Varianza)",
            "Error Cuadratico Medio Escalado",
        ]

    def apply(self, data, scaler):
        components = data.shape[1]
        pca = self.get_pca_information(data, components)
        variance = pca.explained_variance_ratio_
        variance_df = pd.DataFrame(variance).T
        trucanted_components = self.trucante_components(components, variance_df)
        pca_retrained = self.get_pca_information(data, trucanted_components)
        transformed_data = self.transform_data(
            pca_retrained, data, trucanted_components
        )
        error = self.get_error(data, transformed_data, pca_retrained)
        self.metadata["transformed_data"] = transformed_data
        self.metadata["transformer"] = pca_retrained
        self.metadata["summary"] = pd.DataFrame(
            [["pca", scaler, trucanted_components, np.round(error, 4)]],
            columns=self.schema,
        )

    def get_pca_information(self, data, components):
        pca = PCA(
            iterated_power="auto",
            n_components=components,
            random_state=17,
            svd_solver="auto",
            tol=1e-3,
            whiten=False,
        )
        pca.fit_transform(data)
        return pca

    def __str__(self):
        return "PCA"
