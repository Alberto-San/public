import pandas as pd
import numpy as np
import requests
import io
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor # Algoritmo LOF
from scipy.special import entr # Entropía de Shannon
from scipy.stats import multivariate_normal # Modelo de distribución gaussiana multivariable
from sklearn.covariance import EllipticEnvelope # Estimación de Covarianza
from scipy.stats import median_abs_deviation # MAD
from scipy.stats import iqr # Interquartile range
import matplotlib

def plotBoxplot(data):
    sns.set(rc={'figure.figsize':(25,9)}) # Tamaño de la figura
    sns.set(style="whitegrid") # Estilo de la figura
    sns.boxplot(data = data, linewidth = 3, palette="Set2", fliersize = 5) # Diagrama Box Plot
    sns.despine(left=True)
    plt.show()

def plotAgainstOriginal(data, data_without_outliers, feature_1, feature_2):
    plt.scatter(data[feature_1], data[feature_2], color="r")
    plt.scatter(data_without_outliers[feature_1], data_without_outliers[feature_2], color='b')
    plt.xlabel(feature_1)
    plt.ylabel(feature_2)
    plt.show()

def getEntropy(data_original, data_without_outlier, outlierMethod, k = 0, show = False):
    min = data_original.min()
    data_original = data_original - min + 0.001
    data_without_outlier = data_without_outlier - min + 0.001
    H_1 = entr(data_original[data_original.columns])
    H_2 = entr(data_without_outlier[data_without_outlier.columns])
    entropia_normalizada_1 = H_1.sum()/data_original.shape[0]
    entropia_normalizada_2 = H_2.sum()/data_without_outlier.shape[0]
    entropia_normalizada_1 = pd.DataFrame(entropia_normalizada_1, index = None, columns = ['Entropia Original'])
    entropia_normalizada_2 = pd.DataFrame(entropia_normalizada_2, index = None, columns = ["Entropia " + outlierMethod])
    Comparacion_entropias = pd.concat([entropia_normalizada_1, entropia_normalizada_2], axis = 1)
    if show:
      print(Comparacion_entropias.head(k))
    return Comparacion_entropias

def scaleData(data):
    scaler = MinMaxScaler(copy=True, feature_range=(0, 1)) # Método MinMax con valores entre 0 y 1
    columns = data.columns
    dataScaled = scaler.fit_transform(data) # Transformación de los nuevos datos con una escala MinMax
    return pd.DataFrame(dataScaled, columns = columns)

class IQR_processing():
  def getQuartils(self, data):
    iqr_ = iqr(data, axis = 0, rng = (25, 75), interpolation = 'midpoint')
    dc = iqr_/2
    q_1 = np.percentile(data, 25, axis = 0, interpolation = 'midpoint')
    q_3 = np.percentile(data, 75, axis = 0, interpolation = 'midpoint')
    lower_whisker = q_1 - 1.5*iqr_
    upper_whisker = q_3 + 1.5*iqr_
    return {
        "dc": dc,
        "lower_whisker": lower_whisker,
        "upper_whisker": upper_whisker,
        "iqr": iqr_
    }

  def boxPlotWithLimits(self, data, quartilsInformation):
      sns.set(rc={'figure.figsize':(25,9)}) # Tamaño de la figura
      sns.set(style="whitegrid") # Estilo de la figura
      sns.boxplot(data = data, linewidth = 3, palette="Set2", fliersize = 5) # Diagrama Box Plot

      for m in range(len(quartilsInformation["iqr"])):
          plt.axhline(quartilsInformation["lower_whisker"][m], color = 'r')
          plt.axhline(quartilsInformation["upper_whisker"][m], color = 'b')

      sns.despine(left=True)
      plt.show()
  
  def getOutliersFromQuartils(self, data, quartils, display_reporting = False):
      data_outliers = [[] for _ in range(len(data.columns))]
      index_data_outliers = [[] for _ in range(len(data.columns))]

      for n in range(data.shape[0]):
          for indexColumn in range(len(data.columns)):
              if data.iloc[n,indexColumn] < quartils["lower_whisker"][indexColumn] or data.iloc[n,indexColumn] > quartils["upper_whisker"][indexColumn] :
                  data_outliers[indexColumn].append(data.iloc[n,indexColumn])
                  index_data_outliers[indexColumn].append(n)

      amount_outliers_per_variable = [len(oulier_data) for oulier_data in data_outliers]
      reporte = pd.DataFrame(amount_outliers_per_variable, index = data.columns, columns = ['Número de Datos Atípicos'])

      if display_reporting:
          display(reporte)

      return {
          "data_outliers": data_outliers,
          "index_data_outliers": index_data_outliers,
          "reporte": reporte
      }

  def innerJoinIndexes(self, metadata_outliers, verbose = False):
      for index in range(len(metadata_outliers["index_data_outliers"])):
          currentListIndexes = set(metadata_outliers["index_data_outliers"][index])
          
          if index == 0:
              localSet = currentListIndexes
          else:
            localSet = currentListIndexes & localSet
          if verbose:
            print("currentListIndexes: {}".format(currentListIndexes))
            print("localSet: {}".format(localSet))
      return list(localSet)

  def pluggingFiltering(self, data, quartils):
      plugging_data = data.copy()
      columns = plugging_data.columns
      
      for indexColumn in range(len(columns)):
          filter_lower = plugging_data[columns[indexColumn]] < quartils["lower_whisker"][indexColumn]
          filter_upper = plugging_data[columns[indexColumn]] > quartils["upper_whisker"][indexColumn]
          lower_condition = np.where(
              filter_lower,
              quartils["lower_whisker"][indexColumn],
              plugging_data[columns[indexColumn]]
          )
          plugging_data[columns[indexColumn]] = np.where(
              filter_upper,
              quartils["upper_whisker"][indexColumn],
              lower_condition
          ) # converts outlier data to whisker value, it does not delete.

      data_without_outlier = plugging_data
      return data_without_outlier

  def removeOutliersQuartils(self, data, metadata_outliers, quartils, method = "drop", show_information = False):
      if method == "drop":
          inner_possitions = self.innerJoinIndexes(metadata_outliers)
          data_without_outlier = data.drop(index=inner_possitions)
      else:
          data_without_outlier = self.pluggingFiltering(data, quartils)

      if show_information:
        print("number of samples with outliers: {}".format(data.shape))
        print("number of samples without outliers: {}".format(data_without_outlier.shape))
      return data_without_outlier

class Z_test():
   def z_test_normalization(self, data):
      mean = np.mean(data)
      std = np.std(data)
      z = (data - mean)/std
      return z

   def z_outlier_processing(self, data, z, deviation_rule=3, k=5, show_report=False):
      columns = z.columns
      data_outliers = [[] for _ in columns]
      data_outliers_indexes = [[] for _ in columns]

      for n in range(z.shape[0]):
         for column_id in range(len(columns)):
            if z.iloc[n,0] > deviation_rule:
               data_outliers[column_id].append(data.iloc[n,column_id])
               data_outliers_indexes[column_id].append(n)

      outlier_length = [len(outlier) for outlier in data_outliers]
      report = pd.DataFrame(outlier_length, index = columns, columns = ['Número de Datos Atípicos'])

      if show_report:
         print(report.head(k))

      return {
         "report": report,
         "data_outliers": data_outliers,
         "index_data_outliers": data_outliers_indexes
      }

   def innerJoinIndexes(self, metadata_outliers, verbose = False):
         for index in range(len(metadata_outliers["index_data_outliers"])):
            currentListIndexes = set(metadata_outliers["index_data_outliers"][index])
            
            if index == 0:
               localSet = currentListIndexes
            else:
               localSet = currentListIndexes & localSet
            if verbose:
               print("currentListIndexes: {}".format(currentListIndexes))
               print("localSet: {}".format(localSet))
         return list(localSet)

   def filterOutlier(self, data, outliers_indexes, verbose = False):
      filter_data = data.drop(outliers_indexes)
      if verbose:
         print("Samples with outliers: {}".format(data.shape))
         print("Samples without outliers: {}".format(filter_data.shape))
      return filter_data

class LOF_processing():
  def processOutliersByLOF(self, data, k=5):
      LOF = LocalOutlierFactor(n_neighbors = k, algorithm = 'auto', contamination = 0.05, metric = 'euclidean') 
      process_data = LOF.fit_predict(data)
      NOF = LOF.negative_outlier_factor_
      radio_outiler = (NOF.max() - NOF)/(NOF.max() - NOF.min())
      ground_truth = np.ones(len(data), dtype = int) 
      n_errors = (process_data != ground_truth).sum()
      return {
          "NOF": NOF,
          "process_data": process_data,
          "radio_outiler": radio_outiler,
          "n_errors": n_errors,
          "ground_truth": ground_truth
      }

  def dummyOutlierAnalysisLOF(self, data, metadata, feature_1, feature_2):
      index = np.where(metadata["process_data"] == metadata["ground_truth"])
      index = np.asarray(index)
      index = np.hstack(index)
      values = data.iloc[index]
      plt.scatter(data[feature_1], data[feature_2], color="r")
      plt.scatter(values[feature_1], values[feature_2], color='b')
      plt.xlabel(feature_1)
      plt.ylabel(feature_2)
      plt.show()

  def filterOutliersLOF(self, data, process_LOF_data, ground_truth):
      pos = np.where(process_LOF_data == ground_truth)
      pos = np.asarray(pos)
      pos = np.hstack(pos)
      filter_data = data.iloc[pos]
      return filter_data

class Z_test_modify():
   def z_test_normalization(self, data):
      median = np.median(data, axis = 0)
      MAD = median_abs_deviation(data)
      z_modify = 0.6745*((data - median)/MAD)
      return z_modify

   def z_outlier_processing(self, data, z, deviation_rule=3, show_report=False):
      columns = z.columns
      data_outliers = [[] for _ in columns]
      data_outliers_indexes = [[] for _ in columns]

      for n in range(z.shape[0]):
         for column_id in range(len(columns)):
            if z.iloc[n,0] > deviation_rule:
               data_outliers[column_id].append(data.iloc[n,column_id])
               data_outliers_indexes[column_id].append(n)

      outlier_length = [len(outlier) for outlier in data_outliers]
      report = pd.DataFrame(outlier_length, index = columns, columns = ['Número de Datos Atípicos'])

      if show_report:
         display(report)

      return {
         "report": report,
         "data_outliers": data_outliers,
         "index_data_outliers": data_outliers_indexes
      }

   def innerJoinIndexes(self, metadata_outliers, verbose = False):
         for index in range(len(metadata_outliers["index_data_outliers"])):
            currentListIndexes = set(metadata_outliers["index_data_outliers"][index])
            
            if index == 0:
               localSet = currentListIndexes
            else:
               localSet = currentListIndexes & localSet
            if verbose:
               print("currentListIndexes: {}".format(currentListIndexes))
               print("localSet: {}".format(localSet))
         return list(localSet)

   def filterOutlier(self, data, outliers_indexes, verbose = False):
      filter_data = data.drop(outliers_indexes)
      if verbose:
         print("Samples with outliers: {}".format(data.shape))
         print("Samples without outliers: {}".format(filter_data.shape))
      return filter_data

def applyLOF(scale_data, feature_1=None, feature_2=None, show_info=False):
  print("LOF")
  # process Outliers By LOF Method
  LOF = LOF_processing()
  metadata_LOF = LOF.processOutliersByLOF(scale_data, k=5)
  data_without_outliers_LOF = LOF.filterOutliersLOF(scale_data, metadata_LOF["process_data"], metadata_LOF["ground_truth"])
  entropyByVariableLOF = getEntropy(scale_data, data_without_outliers_LOF, "LOF")

  if show_info:
    # After Scaling
    print(5*"\n" + "Scale Data Boxplot")
    plotBoxplot(scale_data)
    plt.show()
    # Comparing with original
    print(5*"\n" + "Número de muestras con outilers:", scale_data.shape[0])
    print("Número de muestras sin outilers:", data_without_outliers_LOF.shape[0])
    print("outliers in RED")
    plotAgainstOriginal(scale_data, data_without_outliers_LOF, feature_1, feature_2)
    print(5*"\n" + "Difference between Entropies Using LOF: {}".format((entropyByVariableLOF["Entropia Original"] - entropyByVariableLOF["Entropia " + "LOF"]).sum()))
  return entropyByVariableLOF, data_without_outliers_LOF

def applyIQR(scale_data, feature_1=None, feature_2=None, method="drop", show_info=False):
  print("IQR" + method + "...")
  iqr_ = IQR_processing()
  quartils = iqr_.getQuartils(scale_data)
  metadata_outliers = iqr_.getOutliersFromQuartils(scale_data, quartils, show_info)
  if method=="drop":
    data_without_outliers = iqr_.removeOutliersQuartils(data = scale_data, metadata_outliers = metadata_outliers, quartils = quartils, method="drop", show_information=show_info)
  else:
    data_without_outliers = iqr_.removeOutliersQuartils(data = scale_data, metadata_outliers = metadata_outliers, quartils = quartils, method="plugging", show_information=show_info)
  entropyByVariable = getEntropy(scale_data, data_without_outliers, "IQR " + method)

  if show_info:
    print("BoxPlot with outliers")
    iqr_.boxPlotWithLimits(scale_data, quartils)
    print(5*"\n" + "BoxPlot without outliers")
    iqr_.boxPlotWithLimits(data_without_outliers, quartils)
    # Comparing with original
    print("outliers in RED")
    plotAgainstOriginal(scale_data, data_without_outliers, feature_1, feature_2)
    print(5*"\n" + "Difference between Entropies Using IQR DROP: {}".format((entropyByVariable["Entropia Original"] - entropyByVariable["Entropia " + "IQR " + method]).sum()))
  
  return entropyByVariable, data_without_outliers

def testZ(scale_data, feature_1=None, feature_2=None, show_info=False):
  print("test Z...")
  # Using Z-test
  Z_test_ = Z_test()
  z_normalization = Z_test_.z_test_normalization(scale_data)
  metadata_z_test = Z_test_.z_outlier_processing(scale_data, z_normalization)
  outliers_indexes = Z_test_.innerJoinIndexes(metadata_z_test)
  data_without_outliers_Ztest = Z_test_.filterOutlier(scale_data, outliers_indexes, show_info)
  entropyByVariableZTEST = getEntropy(scale_data, data_without_outliers_Ztest, "Z_TEST METHOD")

  if show_info:
    print(5*"\n" + "Z_TEST METHOD")
    print("Difference between Entropies Using Z_TEST METHOD: {}".format((entropyByVariableZTEST["Entropia Original"] - entropyByVariableZTEST["Entropia " + "Z_TEST METHOD"]).sum()))
    plotAgainstOriginal(scale_data, data_without_outliers_Ztest, feature_1, feature_2)

  return entropyByVariableZTEST, data_without_outliers_Ztest

def testZmodificado(scale_data, feature_1=None, feature_2=None, show_info=False):
  print("test Z modificado...")
  Z_test_m = Z_test_modify()
  z_normalization = Z_test_m.z_test_normalization(scale_data)
  metadata_z_test = Z_test_m.z_outlier_processing(scale_data, z_normalization)
  outliers_indexes = Z_test_m.innerJoinIndexes(metadata_z_test)
  data_without_outliers_Ztest_m = Z_test_m.filterOutlier(scale_data, outliers_indexes, show_info)
  entropyByVariableZTESTM = getEntropy(scale_data, data_without_outliers_Ztest_m, "Z_TEST MOD METHOD")
  print("Difference between Entropies Using Entropia Z_TEST MOD METHOD: {}".format((entropyByVariableZTESTM["Entropia Original"] - entropyByVariableZTESTM["Entropia " + "Z_TEST MOD METHOD"]).sum()))

  if show_info:
    print(5*"\n" + "Entropia Z_TEST MOD METHOD")
    plotAgainstOriginal(scale_data, data_without_outliers_Ztest_m, feature_1, feature_2)

  return entropyByVariableZTESTM, data_without_outliers_Ztest_m


def summaryBestEntropy(scale_data, feature_1=None, feature_2=None, show_info=False):
  entropyByVariableLOF, _ = applyLOF(scale_data, feature_1, feature_2, show_info)
  entropyByVariableDROP, _ = applyIQR(scale_data, feature_1, feature_2, method = "drop", show_info = show_info)
  entropyByVariablePLUGGING, _ = applyIQR(scale_data, feature_1, feature_2, method = "plugging", show_info = show_info)
  entropyByVariableZTEST, _ = testZ(scale_data, feature_1, feature_2, show_info = show_info)
  entropyByVariableZTESTM, _ = testZmodificado(scale_data, feature_1, feature_2, show_info = show_info)

  entropyDF = pd.concat([
      entropyByVariableLOF["Entropia Original"],
      entropyByVariableLOF["Entropia LOF"],
      entropyByVariableDROP["Entropia IQR drop"],
      entropyByVariablePLUGGING["Entropia IQR plugging"],
      entropyByVariableZTEST["Entropia Z_TEST METHOD"],
      entropyByVariableZTESTM["Entropia Z_TEST MOD METHOD"]
  ], axis=1)

  entropyDF2 = entropyDF.copy()
  columns = entropyDF2.columns
  for column in columns:
    if column != "Entropia Original":
      entropyDF2[column] = (
          entropyDF2["Entropia Original"].sum() - 
          entropyDF2[column].sum())
    
  return entropyDF2.drop(columns=["Entropia Original"]).drop_duplicates()
