# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Otros objetivos
# -----------------------------------------------------------------------
import math

# Gráficos
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px


# Métodos estadísticos
# -----------------------------------------------------------------------
from scipy.stats import zscore # para calcular el z-score
from sklearn.neighbors import LocalOutlierFactor # para detectar outliers usando el método LOF
from sklearn.ensemble import IsolationForest # para detectar outliers usando el metodo IF
from sklearn.neighbors import NearestNeighbors # para calcular la epsilon

# Para generar combinaciones de listas
# -----------------------------------------------------------------------
from itertools import product , combinations

# Gestionar warnings
# -----------------------------------------------------------------------
import warnings
warnings.filterwarnings('ignore')

def plot_outliers_univariados(dataframe,tipo_grafica = "b",bins = 20,grafica_size = (15,10),k_bigote = 1.5):
    """
    Genera gráficos para analizar la presencia de valores atípicos en las variables numéricas de un DataFrame.

    Parámetros
    ----------
    dataframe : pd.DataFrame
        El DataFrame que contiene las variables a analizar. Solo se consideran las columnas numéricas.
    tipo_grafica : str, opcional, por defecto 'b'
        Tipo de gráfico a generar:
        - 'b': Boxplot para detectar valores atípicos.
        - 'h': Histplot para observar la distribución de los datos.
    bins : int, opcional, por defecto 20
        Número de bins para el histograma (solo aplicable si `tipo_grafica` es 'h').
    grafica_size : tuple, opcional, por defecto (15, 10)
        Tamaño de la figura para los gráficos generados.
    k_bigote : float, opcional, por defecto 1.5
        Factor para determinar el rango de los bigotes en el boxplot (valores atípicos).

    Retorna
    -------
    None
        No retorna ningún valor, pero muestra una figura con gráficos para cada columna numérica del DataFrame.

    Notas
    -----
    - Si el número de columnas numéricas es impar, se elimina el último subplot para evitar espacios vacíos.
    - El gráfico muestra los valores atípicos en rojo para facilitar su identificación.
    - Los gráficos generados pueden ser boxplots o histogramas, dependiendo del parámetro `tipo_grafica`.

    """

    df_num = dataframe.select_dtypes(include=np.number)

    fig,axes = plt.subplots(nrows= math.ceil(len(df_num.columns)/2),ncols=2,figsize = grafica_size,)
    axes = axes.flat

    for indice, columna in enumerate(df_num.columns):
        if tipo_grafica == "h":
            sns.histplot(   x = columna,
                            data = dataframe,
                            ax = axes[indice],
                            bins=bins)
        elif tipo_grafica == "b":
            sns.boxplot(    x = columna,
                            data = dataframe,
                            ax = axes[indice],
                            whis= k_bigote,
                            flierprops = {"markersize":4, "markerfacecolor": "red"})
        else:
            print("Las opciones para el tipo de gráfica son: 'b' para boxplot o 'h' para histplot")
        
        axes[indice].set_title(f"Distribución {columna}")
        axes[indice].set_xlabel("")
        
    if len(df_num.columns) % 2 != 0:
        fig.delaxes(axes[-1])
    
    plt.tight_layout()

def identificar_outliers_iqr(dataframe,k = 1.5):
    """
    Identifica los valores atípicos (outliers) en las columnas numéricas de un DataFrame utilizando el método del rango intercuartílico (IQR).

    Parámetros
    ----------
    dataframe : pd.DataFrame
        El DataFrame que contiene las variables a analizar. Solo se consideran las columnas numéricas.
    k : float, opcional, por defecto 1.5
        Factor que determina el rango de los límites para identificar outliers.
        - Valores más allá de `Q1 - k*IQR` o `Q3 + k*IQR` se consideran outliers.

    Retorna
    -------
    dict
        Diccionario donde cada clave es el nombre de la columna que contiene outliers,
        y el valor es un DataFrame con las filas que tienen valores atípicos en dicha columna.

    Notas
    -----
    - El método IQR es robusto ante la presencia de valores atípicos, ya que se basa en los cuartiles.
    - El límite superior se calcula como `Q3 + k*IQR` y el límite inferior como `Q1 - k*IQR`.
    - Si no se encuentran outliers en una columna, dicha columna no se incluye en el diccionario resultante.

    Ejemplos
    --------
    >>> outliers = identificar_outliers_iqr(df, k=1.5)
    >>> print(outliers)
    """

    df_num = dataframe.select_dtypes(include=np.number)
    dictio_outliers = {}
    for columna in df_num.columns:
        Q1 , Q3 = np.nanpercentile(dataframe[columna],(25,75))
        iqr = Q3 - Q1

        limite_superior = Q3 + (iqr * k)
        limite_inferior = Q1 - (iqr * k)

        condicion_sup = dataframe[columna] > limite_superior
        condicion_inf = dataframe[columna] < limite_inferior

        df_outliers = dataframe[condicion_inf |condicion_sup]
        print(f"La columna {columna.upper()} tiene {df_outliers.shape[0]} outliers entre el total de {dataframe.shape[0]} datos, es decir un {(df_outliers.shape[0]/dataframe.shape[0])*100}%")
        if not df_outliers.empty:
            dictio_outliers[columna] = df_outliers
    
    return dictio_outliers

def identificar_outliers_z(dataframe, limite_desviaciones =3):
    """
    Identifica los valores atípicos (outliers) en las columnas numéricas de un DataFrame utilizando el método del Z-score.

    Parámetros
    ----------
    dataframe : pd.DataFrame
        El DataFrame que contiene las variables a analizar. Solo se consideran las columnas numéricas.
    limite_desviaciones : float, opcional, por defecto 3
        Límite del Z-score para identificar valores atípicos.
        - Valores con Z-score mayor o igual al límite especificado se consideran outliers.

    Retorna
    -------
    dict
        Diccionario donde cada clave es el nombre de la columna que contiene outliers,
        y el valor es un DataFrame con las filas que tienen valores atípicos en dicha columna.

    Notas
    -----
    - El Z-score mide cuántas desviaciones estándar está un valor por encima o por debajo de la media.
    - Los valores con un Z-score absoluto mayor o igual al `limite_desviaciones` se clasifican como outliers.
    - Si no se encuentran outliers en una columna, dicha columna no se incluye en el diccionario resultante.

    Ejemplos
    --------
    >>> outliers = identificar_outliers_z(df, limite_desviaciones=3)
    >>> print(outliers)
    """

    df_num = dataframe.select_dtypes(include=np.number)
    diccionario_outliers = {}
    for columna in df_num.columns:
        condicion_zscore = abs(zscore(dataframe[columna])) >= limite_desviaciones
        df_outliers = dataframe[condicion_zscore]

        print(f"La cantidad de ooutliers para la columna {columna.upper()} es {df_outliers.shape[0]}")

        if not df_outliers.empty:
            diccionario_outliers[columna] = df_outliers
    
    return diccionario_outliers

def visualizar_outliers_bivariados(dataframe, vr, tamano_grafica = (20, 15)):

    df_num = dataframe.select_dtypes(include=np.number)
    num_cols = len(df_num.columns)
    num_filas = math.ceil(num_cols / 2)
    fig, axes = plt.subplots(num_filas, 2, figsize=tamano_grafica)
    axes = axes.flat

    for indice, columna in enumerate(df_num.columns):
        if columna == vr:
            fig.delaxes(axes[indice])
        else:
            sns.scatterplot(x = vr, 
                            y = columna, 
                            data = dataframe,
                            ax = axes[indice])
            
            axes[indice].set_title(columna)
            axes[indice].set(xlabel=None, ylabel = None)

        plt.tight_layout()
