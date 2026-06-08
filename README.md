# ☕ Super Projeto Café
 
> Análise Exploratória e Classificação Probabilística de Espécies de Café (Arabica vs. Robusta)
 
---
 
## 📌 Sobre o Projeto
 
Este projeto realiza uma análise estatística aprofundada sobre um dataset de qualidade de café, com foco na discriminação entre as espécies **Arabica** e **Robusta** a partir de atributos sensoriais. O trabalho é composto por duas entregas principais:
 
- **Notebook de Análise Exploratória** (`analise.ipynb`): EDA com tratamento de nulos e visualizações estatísticas.
- **Dashboard Interativo** (`dashboard1.py`): Aplicação web em Streamlit com simulador de classificação em tempo real usando três algoritmos distintos.
---
 
## 🧠 Funcionalidades
 
### Seção 1 — Análise dos Dados
- Métricas gerais do dataset (total de lotes, proporção por espécie)
- Distribuição de frequência e boxplot por atributo sensorial
- Matriz de correlação linear entre as features
- Dispersão bivariada interativa entre pares de atributos
### Seção 2 — Classificação Probabilística
- **Teorema de Bayes** com hipótese gaussiana: calcula probabilidades a posteriori para cada espécie
- **Árvore de Decisão** (max_depth=4, class_weight balanced)
- **K-Nearest Neighbors** (k=3)
- Simulador com sliders interativos e cenários pré-definidos
- Gráfico comparativo unificado dos três métodos
---
 
## 📂 Estrutura do Repositório
 
```
Super_Projeto_Cafe/
│
├── analise.ipynb                  # Notebook de EDA
├── dashboard1.py                  # Dashboard Streamlit
├── merged_data_cleaned.csv        # Dataset tratado
├── grafico_posteriori.png         # Visualização das distribuições a posteriori
├── grafico_priors.png             # Visualização das probabilidades a priori
├── grafico_verossimilhancas.png   # Visualização das verossimilhanças
├── .streamlit/                    # Configurações do Streamlit
└── .gitignore
```
 
---
 
## 🚀 Como Executar
 
### Pré-requisitos
 
- Python 3.8+
- pip
### Instalação das dependências
 
```bash
pip install streamlit pandas numpy plotly scikit-learn scipy
```
 
### Rodando o dashboard
 
```bash
streamlit run dashboard1.py
```
 
O dashboard estará disponível em `http://localhost:8501`.
 
---
 
## 📊 Dataset
 
O arquivo `merged_data_cleaned.csv` contém avaliações sensoriais de lotes de café. As principais features utilizadas nos modelos são:
 
| Feature   | Descrição                          |
|-----------|------------------------------------|
| `Aroma`   | Nota de aroma do lote (0–10)       |
| `Flavor`  | Nota de sabor (0–10)               |
| `Acidity` | Nota de acidez (0–10)              |
| `Body`    | Nota de corpo (0–10)               |
| `Species` | Espécie: `Arabica` ou `Robusta`    |
 
---
 
## 🛠️ Tecnologias Utilizadas
 
- **Python**
- **Streamlit** — interface web interativa
- **Pandas / NumPy** — manipulação de dados
- **Plotly** — visualizações interativas
- **Scikit-learn** — modelos de Machine Learning
- **SciPy** — cálculo de verossimilhanças gaussianas
---
