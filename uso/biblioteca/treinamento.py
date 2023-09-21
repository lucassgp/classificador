# sequencia de codigos para utilizar os dados organizados no DataFrame para treinar o modelo de machine learning

# importacoes

# geograficos
import geopandas as gpd

# dados
import pandas as pd

# graficos
import matplotlib.pyplot as plt
import seaborn as sns

# setando o formato dos graficos
sns.set(rc={'figure.figsize':(15,10)})
sns.set_theme(style="whitegrid", palette="pastel")
sns.set_style("ticks")
sns.set_context("notebook")

# ML
import lightgbm as lgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, recall_score

# outros
from IPython.display import display, HTML

# modelo de classificacao
def preparar_dados_para_treino(dados:pd.DataFrame, amostras:gpd.GeoDataFrame, test_size=0.05) -> list:
    '''Organiza a tabela em uma lista com os objetos separados para treinamento e teste'''

    tabela = pd.merge(dados, amostras[['id', 'label']], on='id')
    df_dev = tabela.copy()

    X_dev = df_dev.drop(['label', 'id'],axis=1)
    y_dev = df_dev.label.astype(int)

    X_tr, X_ts, y_tr, y_ts = train_test_split(X_dev, y_dev, test_size=test_size, random_state=61658)
    return [X_tr, X_ts, y_tr, y_ts]

def treinar_modelo_gbm(dados):
    modelo = lgbm.LGBMClassifier()
    modelo.fit(dados[0], dados[2], eval_set=[(dados[1], dados[3])], verbose= False)
    lgbm.plot_importance(modelo)
    plt.show()
    return modelo

def calculate_metrics(y_true, y_pred):
    confusion = confusion_matrix(y_true, y_pred)
    support = confusion.sum(axis=1)

    tn, fp, fn, tp = confusion.ravel()
    metrics_data = {
        "Sensibilidade": (tp/(tp+fn)),
        "Especificidade": (tn/(fp+tn)),
        "Acurácia": ((tp+tn)/(tp+tn+fp+fn)),
        "Precisão": (tp/(tp+fp)),
        "Amostras classe 0": support[0],
        "Amostras classe 1": support[1],
        "Amostras total": support.sum(),
    }

    df = pd.DataFrame(metrics_data, index=[1])
    df = df.round({'Sensibilidade': 3, 'Especificidade': 3,'Acurácia':3, 'Precisão': 3,
                   'Amostras classe 0':0, 'Amostras classe 1':0, 'Amostras total':0})
    df = df.astype(str).T.reset_index(drop=False)
    df.columns = ['Métrica', 'Valor']
    return df

def metricas(modelo, X_ts, y_ts):
    '''
    funcao para apresentar metricas do modelo recem treinado
        - titulo: titulo do modelo
        - modelo: modelo recem treinado
        - X_ts: variaveis preditoras de teste
        - y_ts: variavel dependente de teste
    '''
    preds = modelo.predict(X_ts)

    print('Validação:')
    display(HTML(calculate_metrics(y_ts, preds).to_html()))

    # graficos

    # matriz de confusao
    sns.heatmap(confusion_matrix(y_ts, preds), annot=True, fmt=".0f", cmap="viridis")
    plt.title(f'Matriz de confusão')
    plt.show()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_ts, preds) # calcula a ROC

    # para a linha diagonal da ROC
    x = [min(fpr), max(fpr)]
    y = [min(tpr), max(tpr)]

    # plotando o gráfico
    sns.lineplot(x=fpr,y=tpr) # plota a ROC
    sns.lineplot(x=x, y=y, color='gray', linestyle='dashdot') # plota a linha diagonal
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.suptitle(f'Curva ROC\nAUC: {round(roc_auc_score(y_ts, preds), 3)}') # calcula a AUC
    plt.show()




