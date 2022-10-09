from common import *
from dimensionalReductionCommon import Dimensional_Reduction_Common


class KernelPCA_(Dimensional_Reduction_Common):
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
        kernel_pca = self.get_kernel_pca_information(data, components)
        lambdas = self.getNormalizeLambdas(kernel_pca)
        trucanted_components = self.trucante_components(components, lambdas)
        kpca_retrained = self.get_kernel_pca_information(data, trucanted_components)
        transformed_data = self.transform_data(
            kpca_retrained, data, trucanted_components
        )
        error = self.get_error(data, transformed_data, kpca_retrained)
        self.metadata["transformed_data"] = transformed_data
        self.metadata["transformer"] = kpca_retrained
        self.metadata["summary"] = pd.DataFrame(
            [["Kernel PCA", scaler, trucanted_components, np.round(error, 4)]],
            columns=self.schema,
        )

    def getNormalizeLambdas(self, kernel_pca):
        lambdas_kpca = kernel_pca.lambdas_
        lambdas_kpca = lambdas_kpca / np.sum(lambdas_kpca)
        lambdas_kpca = pd.DataFrame([lambdas_kpca])
        return lambdas_kpca

    def get_kernel_pca_information(self, data, components):
        kpca = KernelPCA(
            n_components=components,
            kernel="cosine",
            fit_inverse_transform=True,
            random_state=17,
            n_jobs=-1,
        )
        kpca.fit(data)
        return kpca

    def __str__(self):
        return "KPCA"
