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

def explorar_outliers_if(dataframe, df_rellenar, var_dependiente, indice_contaminacion=[0.01, 0.05, 0.1], estimadores=1000, colores={-1: "red", 1: "grey"}, grafica_size = (20, 15)):
        """
        Detecta outliers en un DataFrame utilizando el algoritmo Isolation Forest y visualiza los resultados.

        Params:
            - var_dependiente : str. Nombre de la variable dependiente que se usará en los gráficos de dispersión.
        
            - indice_contaminacion : list of float, opcional. Lista de valores de contaminación a usar en el algoritmo Isolation Forest. La contaminación representa
            la proporción de outliers esperados en el conjunto de datos. Por defecto es [0.01, 0.05, 0.1].
        
            - estimadores : int, opcional. Número de estimadores (árboles) a utilizar en el algoritmo Isolation Forest. Por defecto es 1000.
        
            - colores : dict, opcional. Diccionario que asigna colores a los valores de predicción de outliers del algoritmo Isolation Forest.
            Por defecto, los outliers se muestran en rojo y los inliers en gris ({-1: "red", 1: "grey"}).
        
        Returns:
            Esta función no retorna ningún valor, pero crea y muestra gráficos de dispersión que visualizan los outliers
        detectados para cada valor de contaminación especificado.
        """
        df_num = dataframe.select_dtypes(include=np.number)
        df_if = df_rellenar.copy()

        col_numericas = df_num.columns.to_list()

        num_filas = math.ceil(len(col_numericas) / 2)
        for contaminacion in indice_contaminacion: 
            
            ifo = IsolationForest(random_state=42, 
                                n_estimators=estimadores, 
                                contamination=contaminacion,
                                max_samples="auto",  
                                n_jobs=-1)
            ifo.fit(dataframe[col_numericas])
            prediccion_ifo = ifo.predict(dataframe[col_numericas])
            

            fig, axes = plt.subplots(num_filas, 2, figsize=grafica_size) 
            axes = axes.flat
            for indice, columna in enumerate(col_numericas):
                df_if[f"outlier_{contaminacion}_{columna}_isoforest"] = prediccion_ifo
                if columna == var_dependiente:
                    fig.delaxes(axes[indice])

                else:
                    # Visualizar los outliers en un gráfico
                    sns.scatterplot(x=var_dependiente, 
                                    y=columna, 
                                    data=df_if,
                                    hue=f"outlier_{contaminacion}_{columna}_isoforest", 
                                    palette=colores, 
                                    style=f"outlier_{contaminacion}_{columna}_isoforest", 
                                    size=2,
                                    ax=axes[indice])
                    
                    axes[indice].set_title(f"Contaminación = {contaminacion} y columna {columna.upper()}")
                    plt.tight_layout()
                
            print(f"se ha hecho outlier_{contaminacion}_{columna}_isoforest")           
            if len(col_numericas) % 2 != 0:
                fig.delaxes(axes[-1])
            
        print("Se devuelve df Modificado")
        return df_if

def explorar_outliers_lof(dataframe,df_rellenar, var_dependiente, indice_contaminacion=[0.01, 0.05, 0.1], vecinos=[600, 1200, 1500, 2000], colores={-1: "red", 1: "grey"}, grafica_size = (20, 15)):
    """
    Detecta outliers en un DataFrame utilizando el algoritmo Local Outlier Factor (LOF) y visualiza los resultados.

    Params:
        - var_dependiente : str. Nombre de la variable dependiente que se usará en los gráficos de dispersión.
        
        - indice_contaminacion : list of float, opcional. Lista de valores de contaminación a usar en el algoritmo LOF. La contaminación representa
        la proporción de outliers esperados en el conjunto de datos. Por defecto es [0.01, 0.05, 0.1].
        
        - vecinos : list of int, opcional. Lista de números de vecinos a usar en el algoritmo LOF. Por defecto es [600, 1200, 1500, 2000].
        
        - colores : dict, opcional. Diccionario que asigna colores a los valores de predicción de outliers del algoritmo LOF.
        Por defecto, los outliers se muestran en rojo y los inliers en gris ({-1: "red", 1: "grey"}).

    Returns:
        
        Esta función no retorna ningún valor, pero crea y muestra gráficos de dispersión que visualizan los outliers
        detectados para cada combinación de vecinos y nivel de contaminación especificado.
    """
    df_num = dataframe.select_dtypes(include=np.number)
    # Hacemos una copia del dataframe original para no hacer modificaciones sobre el original
    df_lof = df_rellenar.copy()
        
    # Extraemos las columnas numéricas 
    col_numericas = df_num.columns.to_list()

    # Generamos todas las posibles combinaciones entre los vecinos y el nivel de contaminación
    combinaciones = list(product(vecinos, indice_contaminacion))

    # Iteramos por cada posible combinación
    for combinacion in combinaciones:
        # Aplicar LOF con un número de vecinos y varias tasas de contaminación
        clf = LocalOutlierFactor(n_neighbors=combinacion[0], contamination=combinacion[1])
        y_pred = clf.fit_predict(dataframe[col_numericas])

        num_filas = math.ceil(len(col_numericas) / 2)

        fig, axes = plt.subplots(num_filas, 2, figsize=grafica_size)
        axes = axes.flat

        # Asegurar que la variable dependiente no está en las columnas numéricas
        if var_dependiente in col_numericas:
            col_numericas.remove(var_dependiente)

        for indice, columna in enumerate(col_numericas):
            # Agregar la predicción de outliers al DataFrame
            df_lof[f"outlier_{combinacion[1]}_{columna}_{combinacion[0]}vecinos_lof"] = y_pred
            # Visualizar los outliers en un gráfico
            sns.scatterplot(x=var_dependiente, 
                            y=columna, 
                            data=df_lof,
                            hue=f"outlier_{combinacion[1]}_{columna}_{combinacion[0]}vecinos_lof", 
                            palette=colores, 
                            style=f"outlier_{combinacion[1]}_{columna}_{combinacion[0]}vecinos_lof", 
                            size=2,
                            ax=axes[indice])
                
            axes[indice].set_title(f"Contaminación = {combinacion[1]} y vecinos {combinacion[0]} y columna {columna.upper()}")
            print(f"se ha hecho outlier_{combinacion[1]}_{columna}_lof")
        plt.tight_layout()

        if len(col_numericas) % 2 != 0:
            fig.delaxes(axes[-1])

        plt.show()
    print("Se devuelve df Modificado")
    return df_lof