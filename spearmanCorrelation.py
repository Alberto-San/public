from public.common import *


class Spearman_Correlation:
    def corr_spearman(self, Features):
        spearman = []
        valor_p = []
        Data_cemento = Features.values
        Data_cemento = np.asarray(Data_cemento)

        for _, n in enumerate(np.arange(0, Data_cemento.shape[1])):
            for _, m in enumerate(np.arange(0, Data_cemento.shape[1])):
                s_valor, p_valor = spearmanr(Data_cemento[:, n], Data_cemento[:, m])
                spearman.append(s_valor)
                valor_p.append(p_valor)

        spearman = np.asarray(spearman)
        spearman_r = spearman.reshape(Data_cemento.shape[1], Data_cemento.shape[1])
        valor_p = np.asarray(valor_p)
        p_value = valor_p.reshape(Data_cemento.shape[1], Data_cemento.shape[1])
        spearman_r, p_value = pd.DataFrame(
            spearman_r, index=Features.columns, columns=Features.columns
        ), pd.DataFrame(p_value, index=Features.columns, columns=Features.columns)
        return spearman_r, p_value

    def correlation_analysis(self, df_no_scale):
        spearman_r, p_value_spearman = self.corr_spearman(df_no_scale)
        columns = p_value_spearman.columns

        traverse = {}
        for row_index in range(p_value_spearman.shape[0]):
            traverse[row_index] = {}
            for column_index in range(p_value_spearman.shape[0]):
                traverse[row_index][column_index] = True

        for row_index in range(p_value_spearman.shape[0]):
            for column_index in range(p_value_spearman.shape[0]):
                if (
                    p_value_spearman[columns[column_index]][row_index] < 0.05
                    and row_index != column_index
                    and (
                        spearman_r[columns[column_index]][row_index] > 0.8
                        or spearman_r[columns[column_index]][row_index] < -0.8
                    )
                    and traverse[row_index][column_index] == True
                    and traverse[column_index][row_index] == True
                ):
                    print(
                        "({}, {}) (pvalue={:.3f}, corr={:.3f})".format(
                            columns[row_index],
                            columns[column_index],
                            p_value_spearman[columns[column_index]][row_index],
                            spearman_r[columns[column_index]][row_index],
                        )
                    )
                    traverse[row_index][column_index] = False
