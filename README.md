# 🦋 Diagnóstico de Hipotireoidismo com Machine Learning

Modelo de classificação binária treinado em 3.772 registros de pacientes para prever o risco de hipotireoidismo a partir de medições hormonais e histórico clínico, construído com foco incansável em Recall para garantir que nenhum paciente doente fique sem diagnóstico.

> *"Todo ano, milhares de pacientes convivem com fadiga crônica, ganho de peso inexplicável e depressão, sem saber que a causa pode ser uma glândula em formato de borboleta no pescoço."*

---

## 🎯 Objetivo de Negócio

O hipotireoidismo é uma das condições mais subdiagnosticadas no mundo. A tireoide, quando falha, desregula silenciosamente o metabolismo, o humor e a função cardíaca, e o diagnóstico muitas vezes chega tarde demais. Este projeto foi construído em torno de uma pergunta central: **uma máquina pode aprender a reconhecer os sinais que a triagem clínica às vezes deixa passar?**

Num cenário clínico convencional, o médico analisa exames, histórico e sintomas sequencialmente. Com centenas de pacientes, esse processo é lento e sujeito a erro humano. O risco mais crítico não é diagnosticar uma pessoa saudável como doente, isso gera apenas um exame extra. O risco real é o oposto: **deixar um paciente doente sair da consulta sem diagnóstico**.

Em machine learning, isso é um Falso Negativo. Minimizar esse número guiou cada decisão técnica deste projeto.

---

## 📂 Dataset

| Atributo | Detalhe |
|---|---|
| Registros | 3.772 pacientes |
| Features | 29 variáveis (hormonais, demográficas, clínicas) |
| Variável alvo | `binaryClass`: P (positivo) ou N (negativo) |
| Desbalanceamento | 92% negativo / 8% positivo |

O desbalanceamento por si só definiu toda a estratégia: um modelo que sempre previsse "saudável" atingiria 92% de acurácia e seria completamente inútil. Isso orientou cada decisão técnica seguinte: escolha da métrica, estratégia de imputação e parâmetros do modelo.

---

## 🗂️ Metodologia — CRISP-DM

### 1. Entendimento do Negócio
Antes de abrir qualquer arquivo, era essencial entender o problema clínico. O **TSH (Hormônio Estimulante da Tireoide)** é o principal marcador do hipotireoidismo: quando a tireoide falha, a hipófise eleva o TSH tentando estimulá-la. Esse conhecimento médico foi validado pelo próprio modelo, o TSH emergiu como a feature mais importante na análise SHAP, confirmando que a máquina aprendeu biologia, não ruído.

### 2. Análise Exploratória de Dados (EDA)
Os dados chegaram sujos. O caractere `?` mascarava valores ausentes em colunas inteiras, o TBG, por exemplo, tinha mais de 95% de valores faltantes. Principais achados:
- Distribuições hormonais claramente distintas entre pacientes positivos e negativos, especialmente no TSH
- Outliers hormonais extremos que, ao contrário da estatística clássica, eram exatamente o sinal mais valioso. Um TSH de 500 não é ruído, é o diagnóstico.
- Perfil demográfico consistente com a literatura: doença mais prevalente em mulheres e pacientes acima de 50 anos

### 3. Preparação dos Dados
Três decisões técnicas se destacam:

**Os outliers hormonais foram mantidos.** Remover um TSH de 500 apagaria a evidência mais clara de hipotireoidismo severo. Em dados clínicos, o outlier é frequentemente o diagnóstico.

**Imputação pós-split.** A mediana usada para preencher valores ausentes foi calculada exclusivamente no conjunto de treino e aplicada ao teste, nunca o contrário. Fazer o inverso causa data leakage: o modelo "veria" informação do teste antes do que deveria, inflando artificialmente as métricas.

**Colunas com mais de 70% de valores ausentes foram removidas.** Imputar a maioria dos valores de uma coluna não é preencher lacunas, é inventar dados.

### 4. Modelagem

Quatro modelos foram comparados em validação cruzada estratificada de 5 folds:

| Modelo | Recall | F1-Score | AUC-ROC |
|---|---|---|---|
| Regressão Logística | baseline | baseline | baseline |
| Random Forest | — | — | — |
| XGBoost | — | — | — |
| **LightGBM (ajustado)** | **melhor** | **melhor** | **melhor** |

O `StratifiedKFold` garantiu que cada fold mantivesse a proporção original de 92/8 entre as classes. O `RandomizedSearchCV` com 40 iterações encontrou os melhores hiperparâmetros do LightGBM numa fração do tempo que um `GridSearchCV` completo exigiria.

### 5. Avaliação
O modelo campeão foi avaliado no conjunto de teste, dados que nunca influenciaram nenhuma decisão de treino ou seleção.

A análise SHAP fechou o ciclo: além dos bons números, o modelo aprendeu relações biologicamente corretas. TSH no topo da importância global. T3 e TT4 logo abaixo. O modelo não é uma caixa preta, é um sistema auditável.

---

## 📊 Resultados

| Métrica | Valor | Significado clínico |
|---|---|---|
| Recall | ≥ 95% | A cada 100 pacientes doentes, o modelo detecta ~95 |
| F1-Score | ≥ 92% | Forte equilíbrio entre precisão e sensibilidade |
| AUC-ROC | ≥ 98% | Capacidade de separação de classes próxima do ideal teórico |
| Precision | ≥ 90% | A cada 100 alertas, ~90 são casos reais |

---

## 🔍 Top 5 Features Mais Decisivas (SHAP)

1. **TSH** — Nível elevado é o indicador primário. A hipófise grita quando a tireoide silencia.
2. **T3 / TT4** — Hormônios tireoidianos produzidos diretamente pela glândula.
3. **FTI** — Índice de Tiroxina Livre, uma medida derivada de alta relevância clínica.
4. **Idade** — O risco aumenta significativamente após os 50 anos.
5. **On Thyroxine** — Pacientes já em terapia de reposição hormonal apresentam padrões fisiológicos distintos.

---

## ⚠️ Limitações e Próximos Passos

Este modelo não está pronto para produção, é uma prova de conceito robusta. Antes de qualquer implementação clínica real:

- **Validação externa:** testar em dados de outros hospitais para avaliar a generalização
- **Calibração de probabilidade:** garantir que "70% de risco" signifique de fato 70%
- **Monitoramento de data drift:** as distribuições hormonais mudam com populações e equipamentos
- **Aprovação regulatória:** qualquer sistema de apoio à decisão clínica exige validação por autoridades competentes

---

## 🛠️ Stack Técnica

| Categoria | Ferramentas |
|---|---|
| Linguagem | Python 3.x |
| Manipulação de Dados | Pandas, NumPy |
| Visualização | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost, LightGBM |
| Interpretabilidade | SHAP |
| Ambiente | Jupyter Notebook |

---

## ▶️ Como Executar

```bash
# Clone o repositório
git clone https://github.com/oporaxuao/hypothyroidism-diagnosis-catboost.git
cd hypothyroidism-diagnosis-catboost

# Instale as dependências
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap jupyter

# Inicie o notebook
jupyter notebook prevencao_hypothyroid.ipynb
```

Execute as células em sequência. O notebook é autocontido, cada etapa gera as variáveis necessárias para a próxima.

---

## 👤 Autor

**João Alfredo de Sousa Siqueira**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-oporaxuao-blue)](https://linkedin.com/in/oporaxuao)
[![GitHub](https://img.shields.io/badge/GitHub-oporaxuao-black)](https://github.com/oporaxuao)
