import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Dashboard de Análise & Classificação de Café",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #FDFBF7;
    }
    .block-container {
        padding-top: 2rem !important;    
        padding-bottom: 2rem !important;
    }
    
    .stTabs [data-baseweb="tab"] p {
        font-size: 1.1rem;    
        font-weight: 600;    
        text-shadow: 0 1px 3px rgba(139, 111, 71, 0.2);
    }
    
    /* Customização de Cards */
    .metric-card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 5px 12px rgba(139, 111, 71, 0.18);
        border-left: 5px solid #8C6D53;
        margin-bottom: 1rem;
    }
    .metric-title {
        color: #8C6D53;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #3B2F2F;
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.2rem;
    }
    
    /* Títulos de Seções */
    .section-header {
        color: #8C6D53;
        font-weight: 700;
        border-bottom: 2px solid #E6DCD2;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        
    }
    
    .objective-box {
        background-color: #F4EFEA;
        border-left: 4px solid #A67B5B;
        padding: 0.8rem 1.2rem;
        border-radius: 4px;
        font-size: 1rem;
        color: #5C4D43;
        margin-bottom: 1rem;
        
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def carregar_e_limpar_dados():
    try:
        df = pd.read_csv('merged_data_cleaned.csv')
    except FileNotFoundError:
        st.error("Arquivo 'merged_data_cleaned.csv' não encontrado. Certifique-se de que ele está no mesmo diretório.")
        return pd.DataFrame()
    
    features_projeto = ['Aroma', 'Flavor', 'Acidity', 'Body']
    for col in features_projeto:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
            
    return df

df_cafe = carregar_e_limpar_dados()

if df_cafe.empty:
    st.stop()

FEATURES = ['Aroma', 'Flavor', 'Acidity', 'Body']
X = df_cafe[FEATURES]
y = df_cafe['Species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

clf_dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
clf_dt.fit(X_train, y_train)

clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(X_train, y_train)

bayes_params = {}
for classe in df_cafe['Species'].unique():
    df_classe = df_cafe[df_cafe['Species'] == classe]
    bayes_params[classe] = {
        'prior': len(df_classe) / len(df_cafe),
        'stats': {feat: {'mean': df_classe[feat].mean(), 'std': df_classe[feat].std()} for feat in FEATURES}
    }

st.markdown('<p style="font-size: 2.5rem; font-weight: 800; color: #6F4E40; margin-bottom: 0px; text-shadow: 0 1px 3px rgba(139, 111, 71, 0.5); "> Dashboard de Análise & Classificação de Café ☕</p>', unsafe_allow_html=True)
st.markdown('<p style="font-size: 1.2rem; color: #8C6D53; margin-top: 0px;">Exploração Estatística Avançada e Predição Probabilística (Arabica vs. Robusta)</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Seção 1: Análise dos Dados", "Seção 2: Classificação Probabilística"])

with tab1:
    st.markdown('<h2 class="section-header" style="font-size: 1.5rem; font-weight: 700; text-shadow: 0 1px 3px rgba(139, 111, 71, 0.5); " >Histórico e Padrões Estatísticos do Dataset</h2>', unsafe_allow_html=True)
    
    total_amostras = len(df_cafe)
    pct_arabica = (df_cafe['Species'] == 'Arabica').mean() * 100
    pct_robusta = (df_cafe['Species'] == 'Robusta').mean() * 100
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total de Lotes Analisados</div><div class="metric-value">{total_amostras}</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Proporção Arabica</div><div class="metric-value">{pct_arabica:.2f}%</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Proporção Robusta</div><div class="metric-value">{pct_robusta:.2f}%</div></div>', unsafe_allow_html=True)
        
    st.markdown('#### 1. Distribuição de Atributos por Espécie')
    st.markdown('<div class="objective-box"><b>Objetivo Analítico:</b> Avaliar visualmente o comportamento das notas sensoriais de cada espécie, identificando sobreposições ou se alguma variável isolada atua como um bom separador linear para a classificação.</div>', unsafe_allow_html=True)
    
    atributo_selecionado = st.selectbox("Escolha o atributo para analisar a distribuição:", FEATURES)
    fig_dist = px.histogram(
        df_cafe, x=atributo_selecionado, color="Species", marginal="box",
        barmode="overlay", color_discrete_sequence=['#8B5A2B', '#CDA47E'],
        title=f"Distribuição de Frequência e Boxplot para: {atributo_selecionado}"
    )
    fig_dist.update_layout(plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', font_color='#3B2F2F')
    st.plotly_chart(fig_dist, use_container_width=True)
    
    col_g2, col_g3 = st.columns(2)
    
    with col_g2:
        st.markdown('#### 2. Matriz de Correlação Linear')
        st.markdown('<div class="objective-box"><b>Objetivo Analítico:</b> Identificar o grau de associação e redundância de informação entre as variáveis físico-sensoriais do café.</div>', unsafe_allow_html=True)
        
        corr_matrix = df_cafe[FEATURES].corr()
        fig_corr = px.imshow(
            corr_matrix, text_auto='.2f',
            color_continuous_scale=[[0, '#FDFBF7'], [0.5, '#CDA47E'], [1, '#6F4E37']],
            title="Matriz de Correlação das Features Selecionadas"
        )
        fig_corr.update_layout(font_color='#3B2F2F')
        st.plotly_chart(fig_corr, use_container_width=True)
        
    with col_g3:
        st.markdown('#### 3. Dispersão Bivariada das Espécies')
        st.markdown('<div class="objective-box"><b>Objetivo Analítico:</b> Mapear clusters naturais formados pela combinação de dois atributos simultâneos, evidenciando as fronteiras de decisão.</div>', unsafe_allow_html=True)
        
        feat_x = st.selectbox("Eixo X:", FEATURES, index=0)
        feat_y = st.selectbox("Eixo Y:", FEATURES, index=1)
        
        fig_scat = px.scatter(
            df_cafe, x=feat_x, y=feat_y, color="Species",
            color_discrete_sequence=['#8B5A2B', '#D2B48C'],
            title=f"Dispersão Interativa: {feat_x} vs {feat_y}"
        )
        fig_scat.update_layout(plot_bgcolor='white', paper_bgcolor='rgba(0,0,0,0)', font_color='#3B2F2F')
        st.plotly_chart(fig_scat, use_container_width=True)

with tab2:
    st.markdown('<h2 class="section-header" style="font-size: 1.5rem; font-weight: 700;">Simulador Epistêmico e Comparação de Algoritmos</h2>', unsafe_allow_html=True)
    
    st.sidebar.markdown('#  Painel de Atributos do Café')
    st.sidebar.markdown('Ajuste as características sensoriais do lote para testar as respostas em tempo real dos modelos.')
    
    cenarios_preset = {
        "Customizado": None,
        "Café Especial Leve": {"Aroma": 8.0, "Flavor": 7.8, "Acidity": 8.2, "Body": 7.1},
        "Café Robusta Tradicional": {"Aroma": 6.8, "Flavor": 6.5, "Acidity": 6.2, "Body": 7.9},
        "Café Equilibrado Gourmet": {"Aroma": 7.5, "Flavor": 7.6, "Acidity": 7.5, "Body": 7.5},
        "Notas Altas": {"Aroma": 8.7, "Flavor": 8.6, "Acidity": 8.5, "Body": 8.4}
    }
    
    selecao_preset = st.sidebar.selectbox("Carregar Perfil de Cenário:", list(cenarios_preset.keys()))
    
    def_vals = cenarios_preset[selecao_preset]
    
    val_aroma = st.sidebar.slider("Aroma", 5.0, 10.0, def_vals["Aroma"] if def_vals else 7.5, 0.1)
    val_flavor = st.sidebar.slider("Flavor", 5.0, 10.0, def_vals["Flavor"] if def_vals else 7.5, 0.1)
    val_acidity = st.sidebar.slider("Acidity", 5.0, 10.0, def_vals["Acidity"] if def_vals else 7.5, 0.1)
    val_body = st.sidebar.slider("Body", 5.0, 10.0, def_vals["Body"] if def_vals else 7.5, 0.1)
    
    input_usuario = np.array([[val_aroma, val_flavor, val_acidity, val_body]])
    input_dict = {'Aroma': val_aroma, 'Flavor': val_flavor, 'Acidity': val_acidity, 'Body': val_body}
    
    posteriors = {}
    for classe, params in bayes_params.items():
        prior = params['prior']
        likelihood = 1.0
        for feat in FEATURES:
            mean = params['stats'][feat]['mean']
            std = params['stats'][feat]['std']
            if std > 0:
                likelihood *= norm.pdf(input_dict[feat], loc=mean, scale=std)
        posteriors[classe] = prior * likelihood
        
    soma_posteriors = sum(posteriors.values())
    if soma_posteriors > 0:
        for classe in posteriors:
            posteriors[classe] /= soma_posteriors
    else:
        posteriors = {classe: 1.0 / len(posteriors) for classe in posteriors}
        
    pred_dt = clf_dt.predict(input_usuario)[0]
    pred_knn = clf_knn.predict(input_usuario)[0]
    
    col_b1, col_b2 = st.columns([1, 1])
    
    with col_b1:
        st.markdown('#### Cálculo pelo Teorema de Bayes')
        st.markdown('Abaixo está a distribuição de probabilidade exata obtida multiplicando as probabilidades A Priori pelas verossimilhanças calculadas sob a hipótese gaussiana:')
        
        for classe, prob in posteriors.items():
            st.metric(label=f"Probabilidade Posteriori de ser: **{classe}**", value=f"{prob * 100:.4f}%")
            st.progress(float(prob))
            
    with col_b2:
        st.markdown('#### Predição dos Algoritmos Supervisores')
        st.markdown('Classificação direta (Hard Outputs) efetuada pelos outros classificadores concorrentes:')
        
        st.markdown(f"""
        <div style="background-color: white; padding: 1.2rem; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin-bottom: 1rem; border-left: 5px solid #8B5A2B;">
            <span style="color:#7f7f7f; font-size: 0.85rem; font-weight: bold; text-transform: uppercase;">Árvore de Decisão</span>
            <p style="font-size: 1.6rem; font-weight: 700; color: #3B2F2F; margin: 0;">Espécie: <span style="color:#6F4E37;">{pred_dt}</span></p>
        </div>
        
        <div style="background-color: white; padding: 1.2rem; border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); border-left: 5px solid #CDA47E;">
            <span style="color:#7f7f7f; font-size: 0.85rem;font-weight: bold; text-transform: uppercase;">K-Nearest Neighbors (KNN, k=3)</span>
            <p style="font-size: 1.6rem; font-weight: 700; color: #3B2F2F; margin: 0;">Espécie: <span style="color:#6F4E37;">{pred_knn}</span></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('---')
    st.markdown('#### Comparativo Visual Unificado dos Três Métodos')
    st.markdown('<div class="objective-box"><b>Objetivo Analítico:</b> Comparar diretamente o grau de concordância e convergência dos três algoritmos para a amostra simulada corrente. Permite auditar se o viés estatístico de Bayes concorda com os limites geométricos do KNN e as regras lógicas da Árvore.</div>', unsafe_allow_html=True)
    
    score_bayes = posteriors['Arabica']
    score_dt = 1.0 if pred_dt == 'Arabica' else 0.0
    score_knn = 1.0 if pred_knn == 'Arabica' else 0.0
    
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name='Confiança / Predição (Arabica)',
        x=['Teorema de Bayes (Probabilidade)', 'Árvore de Decisão (Binário)', 'KNN (k=3) (Binário)'],
        y=[score_bayes, score_dt, score_knn],
        marker_color=['#6F4E37', '#8B5A2B', '#CDA47E'],
        text=[f"{score_bayes*100:.2f}%", f"{pred_dt}", f"{pred_knn}"],
        textposition='auto'
    ))
    
    fig_comp.update_layout(
        title="Tendência de Classificação Voltada à Espécie Arabica (Alvo Majoritário)",
        yaxis=dict(title='Score / Saída Normalizada (0 a 1)', range=[0, 1.2]),
        plot_bgcolor='white',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_comp, use_container_width=True)