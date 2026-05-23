import json, sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'c:\Users\Turi\Desktop\Data Science I\Data Science 1\Clase06\Copia_de_Unidad_6_a.ipynb'

with open(path, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Edits by cell index -> new source string
edits = {
    0: (
        "import pandas as pd\n"
        "import numpy as np\n"
        "# Google Colab: monta el Drive para acceder a los archivos subidos\n"
        "from google.colab import drive\n"
        "import os\n"
        "drive.mount('/content/gdrive')\n"
        "# Cambia el directorio de trabajo a la raiz de Google Drive\n"
        "os.chdir(\"/content/gdrive/My Drive\")\n"
        "# pingouin: libreria estadistica de alto nivel, usada para tests por pares (pairwise)\n"
        "! pip install pingouin\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "import statsmodels.api as sm\n"
        "from statsmodels.formula.api import ols\n"
        "import statsmodels\n"
        "# scipy.stats: modulo con distribuciones y tests estadisticos (t-test, ANOVA, etc.)\n"
        "from scipy import stats\n"
        "from pingouin import pairwise_ttests\n"
    ),
    2: (
        "# Colab: cambia el directorio activo a MyDrive para que pd.read_csv encuentre los archivos\n"
        "# Los CSV deben estar en la raiz de tu Google Drive (no en subcarpetas)\n"
        "%cd '/content/gdrive/MyDrive'\n"
        "bank = pd.read_csv(\"bank-full.csv\")\n"
        "print(bank.shape)\n"
        "bank.head()\n"
    ),
    4: (
        "# describe(): genera estadisticas descriptivas de todas las columnas numericas\n"
        "# Muestra: count, mean, std, min, percentiles (25%, 50%, 75%), max\n"
        "# Util para detectar escalas, posibles outliers y la distribucion general de los datos\n"
        "bank.describe()\n"
    ),
    6: (
        "# Media muestral: promedio del saldo bancario de los clientes en el dataset\n"
        "# Esta es la estimacion puntual de la media poblacional (mu)\n"
        "# Hipotesis: el valor poblacional conocido del anio pasado era $1341.12\n"
        "# Queremos saber si la diferencia observada es estadisticamente significativa\n"
        "bank['balance'].mean()\n"
    ),
    8: (
        "# T-test de UNA MUESTRA (one-sample t-test):\n"
        "# Compara la media muestral contra un valor poblacional conocido (popmean)\n"
        "# H0 (hipotesis nula): la media poblacional NO cambio -> mu = 1341.12\n"
        "# Ha (hipotesis alternativa): la media cambio -> mu != 1341.12\n"
        "# Devuelve: estadistico t y p-valor (bilateral por defecto)\n"
        "# Si p-valor < 0.05: se rechaza H0 (la diferencia es estadisticamente significativa)\n"
        "stats.ttest_1samp(bank['balance'], popmean=1341.12)\n"
    ),
    10: (
        "# Test UNILATERAL: cuando la hipotesis alternativa es direccional (Ha: mu > mu0)\n"
        "# scipy devuelve el p-valor bilateral, hay que dividirlo por 2 para el test unilateral\n"
        "# Logica: si el estadistico t es positivo Y p/2 < 0.05 -> se rechaza H0 en favor de Ha: mu > mu0\n"
        "T, p = stats.ttest_1samp(bank['balance'], popmean=1341.122)\n"
        "p_value = p/2\n"
        "p_value\n"
    ),
    12: (
        "# T-test de DOS MUESTRAS INDEPENDIENTES (Welch's t-test):\n"
        "# Compara las medias de dos grupos distintos (clientes con y sin prestamo)\n"
        "# H0: mu_loans = mu_no_loans (no hay diferencia en el saldo promedio entre grupos)\n"
        "# Ha: mu_loans != mu_no_loans\n"
        "# equal_var=False: aplica la correccion de Welch (NO asume varianzas iguales entre grupos)\n"
        "#   Esta es la opcion mas robusta cuando no se puede verificar la homogeneidad de varianzas\n"
        "loans = bank[bank.loan==\"yes\"].balance\n"
        "no_loans = bank[bank.loan==\"no\"].balance\n"
        "\n"
        "statistic, pvalue = stats.ttest_ind(loans, no_loans, equal_var=False)\n"
        "print('Estadistico:', round(statistic,2), 'p-valor:', round(pvalue,2))\n"
    ),
    14: (
        "from scipy.stats import t\n"
        "# Intervalo de confianza al 95% para la media del saldo bancario\n"
        "# Formula: IC = ( x_barra - t_crit * s/sqrt(n) , x_barra + t_crit * s/sqrt(n) )\n"
        "# dof (grados de libertad) = n - 1: penalizamos por estimar la std a partir de la muestra\n"
        "# t.ppf(): funcion de percentil de la distribucion t (valor critico)\n"
        "#   Con muestras grandes (n > 30), la distribucion t se aproxima a la normal (Z)\n"
        "# Interpretacion: si repetimos el muestreo muchas veces, el 95% de los intervalos\n"
        "#   construidos de esta forma contendran la media poblacional verdadera\n"
        "m = bank.balance.mean()\n"
        "s = bank.balance.std()\n"
        "dof = len(bank.balance)-1\n"
        "confianza = 0.95\n"
        "t_crit = np.abs(t.ppf((1-confianza)/2, dof))\n"
        "print(t_crit)\n"
        "(m - s*t_crit/np.sqrt(len(bank.balance)), m + s*t_crit/np.sqrt(len(bank.balance)))\n"
    ),
    15: (
        "# Comparacion de medias entre grupos antes de calcular el intervalo de diferencia\n"
        "# Si las medias son muy distintas visualmente, el test estadistico probablemente\n"
        "# confirmara que la diferencia es significativa\n"
        "loans.mean(), no_loans.mean()\n"
    ),
    16: (
        "import numpy as np, statsmodels.stats.api as sms\n"
        "# Intervalo de confianza para la DIFERENCIA de medias entre dos grupos\n"
        "# CompareMeans: encapsula las estadisticas de dos grupos para compararlos\n"
        "# tconfint_diff(): calcula el IC de (mu1 - mu2)\n"
        "# usevar='unequal': correccion de Welch (no asume igualdad de varianzas)\n"
        "# Interpretacion: si el intervalo NO contiene el 0, la diferencia es significativa\n"
        "#   Ej: (-500, -100) -> los clientes con prestamo tienen menor saldo (diferencia negativa)\n"
        "X1, X2 = bank[bank.loan==\"yes\"].balance, bank[bank.loan==\"no\"].balance\n"
        "cm = sms.CompareMeans(sms.DescrStatsW(X1), sms.DescrStatsW(X2))\n"
        "print(cm.tconfint_diff(usevar='unequal'))\n"
    ),
    19: (
        "print(\"Definiendo los simbolos de stock\")\n"
        "# Lista de simbolos bursatiles de las 5 empresas energeticas a analizar:\n"
        "# D=Dominion Energy, EXC=Exelon, NEE=NextEra Energy, SO=Southern Co., DUK=Duke Energy\n"
        "# Cada simbolo corresponde a un archivo CSV con datos historicos de precios y volumen\n"
        "symbol_data_to_load = ['D','EXC','NEE','SO','DUK']\n"
        "list_of_df = []\n"
        "\n"
        "print(\" --- Inicio de Loop --- \")\n"
        "for i in symbol_data_to_load:\n"
        "    print(\"Procesando Simbolo: \" + i)\n"
        "    temp_df = pd.read_csv(i+'.csv', sep=',')\n"
        "    # Feature engineering: convertir volumen a millones para mejor legibilidad\n"
        "    temp_df['Volume_Millions'] = temp_df['Volume'] / 1000000.0\n"
        "    # Agregar columna identificadora para distinguir empresas al concatenar\n"
        "    temp_df['Symbol'] = i\n"
        "    list_of_df.append(temp_df)\n"
        "\n"
        "print(\" --- Completado loop sobre simbolos --- \")\n"
        "# pd.concat(): une todos los DataFrames en uno solo apilando filas (axis=0)\n"
        "agg_df = pd.concat(list_of_df, ignore_index=True)\n"
        "print(agg_df.shape)\n"
        "agg_df.head()\n"
    ),
    20: (
        "# Verificacion: confirma que el DataFrame concatenado contiene las 5 empresas\n"
        "agg_df.Symbol.unique()\n"
    ),
    22: (
        "# Agrupacion por Fecha y Simbolo para calcular la volatilidad relativa promedio diaria\n"
        "# VolStat: estadistico de volatilidad relativa (rango diario / precio de apertura)\n"
        "#   Mide que tan grande fue el movimiento del precio en relacion a su nivel\n"
        "# groupby + mean(): colapsa multiples registros del mismo dia/empresa en uno solo\n"
        "# reset_index(): convierte los indices grupales de vuelta a columnas normales\n"
        "agg_df1 = agg_df[['Date','Symbol','VolStat']].groupby(by=['Date','Symbol']).mean().reset_index()\n"
        "agg_df1['Date'] = pd.to_datetime(agg_df1['Date'])\n"
        "agg_df1.head()\n"
    ),
    23: (
        "import seaborn as sns\n"
        "import matplotlib.pyplot as plt\n"
        "# Grafico de linea temporal de la volatilidad relativa por empresa\n"
        "# hue=Symbol: dibuja una linea de distinto color por cada empresa\n"
        "# Un pico en VolStat indica un dia de alta fluctuacion de precio\n"
        "# Patron tipico: el 'agrupamiento de volatilidad' (epocas de alta vol. se agrupan)\n"
        "plt.figure(figsize=(15,8))\n"
        "sns.lineplot(x=agg_df1.Date, y=agg_df1.VolStat, hue=agg_df1.Symbol)\n"
        "plt.xlabel('Fecha')\n"
        "plt.ylabel('VolStat')\n"
        "plt.title('Comparacion de volatilidad relativa vs Fecha')\n"
    ),
    24: (
        "# Agrupacion por Fecha y Simbolo para calcular el retorno diario promedio\n"
        "# Return: retorno porcentual diario (cambio de precio cierre vs cierre anterior)\n"
        "#   Return > 0: el precio subio | Return < 0: el precio bajo\n"
        "agg_df2 = agg_df[['Date','Symbol','Return']].groupby(by=['Date','Symbol']).mean().reset_index()\n"
        "agg_df2['Date'] = pd.to_datetime(agg_df1['Date'])\n"
        "agg_df2.head()\n"
    ),
    25: (
        "import seaborn as sns\n"
        "import matplotlib.pyplot as plt\n"
        "# Grafico de linea temporal del retorno diario por empresa\n"
        "# Permite comparar como evoluciono el rendimiento de cada accion a lo largo del tiempo\n"
        "# Empresas con retornos mas estables y positivos son mejores candidatas de inversion\n"
        "plt.figure(figsize=(15,8))\n"
        "sns.lineplot(x=agg_df2.Date, y=agg_df2.Return, hue=agg_df2.Symbol)\n"
        "plt.xlabel('Fecha')\n"
        "plt.ylabel('Return')\n"
        "plt.title('Comparacion de retorno vs Fecha')\n"
    ),
    26: (
        "# Panel comparativo: volatilidad y retorno lado a lado\n"
        "# subplot(121): primer grafico (1 fila, 2 columnas, posicion 1)\n"
        "# subplot(122): segundo grafico (posicion 2)\n"
        "# Ver ambos juntos permite analizar la relacion riesgo-retorno:\n"
        "#   alta volatilidad + alto retorno -> accion agresiva (mayor riesgo, mayor ganancia potencial)\n"
        "#   baja volatilidad + retorno estable -> accion defensiva (menor riesgo)\n"
        "plt.figure(figsize=(18,8))\n"
        "plt.subplot(121)\n"
        "sns.lineplot(x=agg_df1.Date, y=agg_df1.VolStat, hue=agg_df1.Symbol)\n"
        "plt.xlabel('Fecha')\n"
        "plt.ylabel('VolStat')\n"
        "plt.title('Comparacion de volatilidad relativa vs Fecha')\n"
        "plt.subplot(122)\n"
        "sns.lineplot(x=agg_df2.Date, y=agg_df2.Return, hue=agg_df2.Symbol)\n"
        "plt.xlabel('Fecha')\n"
        "plt.ylabel('Return')\n"
        "plt.title('Comparacion de retorno vs Fecha')\n"
    ),
}

for idx, new_source in edits.items():
    cells[idx]['source'] = new_source
    print('Edited cell ' + str(idx))

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Guardado correctamente.')
