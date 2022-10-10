from public.common import *
from public.dimensionalReductionCommon import Dimensional_Reduction_Common


class ICA_(Dimensional_Reduction_Common):
    def __init__(self):
        self.metadata = {}
        self.schema = [
            "Metodo",
            "Tipo de Escalador",
            "No Componentes (98% Varianza)",
            "Error Cuadratico Medio Escalado",
        ]

    def apply(self, data, scaler):
        minimal_amount_components = self.get_truncated_components(data)
        ica_ = self.get_ica_information(data, minimal_amount_components)
        transformed_data = self.transform_data(
            ica_, data, minimal_amount_components
        )  # ica_.transform(data)
        error = self.get_error(data, transformed_data, ica_)
        self.metadata["transformed_data"] = transformed_data
        self.metadata["transformer"] = ica_
        self.metadata["summary"] = pd.DataFrame(
            [["ICA", scaler, minimal_amount_components, np.round(error, 4)]],
            columns=self.schema,
        )

    def get_truncated_components(self, data):
        componentes = np.arange(2, data.shape[1], 1)
        ica_error = []

        for _, n in enumerate(componentes):
            ica_ = self.get_ica_information(data, n)
            data_transformed = ica_.transform(data)
            error = self.get_error(data, data_transformed, ica_)
            ica_error.append(error)

        ica_error_array = np.asarray(ica_error)
        minimal_amount_components = list(np.where(ica_error_array < 0.015))[0][0] + 2
        return minimal_amount_components

    def get_ica_information(self, data, components):
        ica_ = FastICA(
            n_components=components,
            fun="logcosh",
            fun_args={"alpha": 1},
            max_iter=1000,
            random_state=17,
        )
        ica_.fit_transform(data)
        return ica_

    def __str__(self):
        return "ICA"
