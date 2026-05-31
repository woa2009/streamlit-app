"""
app.py — Orquestrador da Interface
Regressão Espectral: RLM, Bagging e Subagging

Responsabilidade exclusiva: layout Streamlit, abas e interação com o usuário.
Toda a matemática vem de models.py.
Todos os gráficos e componentes visuais vêm de components.py.

Autor: Wagner Oliveira de Araujo
"""

import streamlit as st
import pandas as pd

from models import parse_matrix, parse_vector, run_full_analysis
from components import (
    APP_CSS,
    HEADER_HTML,
    COR_RLM,
    COR_BAGGING,
    COR_SUBAGGING,
    build_chart,
    build_scatter_chart,
    metric_card_html,
)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Regressão Espectral - Bagging & Subagging",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(APP_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE VALIDAÇÃO DE PARÂMETROS
# ─────────────────────────────────────────────────────────────────────────────

def parse_int_list(text: str) -> list[int]:
    """Converte string de inteiros separados por vírgula em lista validada."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    result = []
    for p in parts:
        v = int(p)
        if v <= 0:
            raise ValueError(f"Valor inválido de m: '{p}'. Use inteiros positivos.")
        result.append(v)
    return result


def parse_float_list(text: str) -> list[float]:
    """Converte string de floats separados por vírgula em lista validada."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    result = []
    for p in parts:
        v = float(p)
        if not (0 < v <= 1):
            raise ValueError(f"Valor inválido de k: '{p}'. Use decimais entre 0 e 1.")
        result.append(v)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TEXTOS DE AJUDA (expanders)
# ─────────────────────────────────────────────────────────────────────────────

_TIPOS_MATRIZ = ["txt", "csv", "xlsx"]
_TIPOS_VETOR  = ["txt", "csv", "xlsx"]

_HELP_MATRIZ = (
    "📐 Formato de matriz — aceita .txt · .csv · .xlsx\n\n"
    "• .txt  → colunas separadas por espaços simples, uma amostra por linha.\n"
    "• .csv  → separador de coluna detectado automaticamente (vírgula ou ponto-e-vírgula).\n"
    "• .xlsx → primeira planilha, sem linha de cabeçalho; cada linha = uma amostra.\n\n"
    "Não misture separadores de coluna e decimal no mesmo arquivo."
)
_HELP_VETOR = (
    "📋 Formato de vetor coluna — aceita .txt · .csv · .xlsx\n\n"
    "• .txt  → um valor por linha, sem separadores adicionais.\n"
    "• .csv  → primeira coluna utilizada; cabeçalho não-numérico ignorado.\n"
    "• .xlsx → primeira coluna da primeira planilha.\n\n"
    "Não utilize vírgulas ou ponto e vírgula em arquivos .txt."
)
_EXP_MATRIZ = (
    "**Formatos aceitos:** `.txt` · `.csv` · `.xlsx`\n\n"
    "| Formato | Regra |\n"
    "|---------|-------|\n"
    "| `.txt`  | Colunas separadas por **espaços simples**, uma amostra por linha |\n"
    "| `.csv`  | Separador de coluna: **vírgula** ou **ponto-e-vírgula** (detectado automaticamente) |\n"
    "| `.xlsx` | Primeira planilha, sem linha de cabeçalho |\n\n"
    "**Exemplo (.txt):**\n```\n0.123 0.456 0.789\n0.321 0.654 0.987\n0.111 0.222 0.333\n```\n\n"
    "**Exemplo (.csv com `;`):**\n```\n0.123;0.456;0.789\n0.321;0.654;0.987\n```"
)
_EXP_VETOR = (
    "**Formatos aceitos:** `.txt` · `.csv` · `.xlsx`\n\n"
    "| Formato | Regra |\n"
    "|---------|-------|\n"
    "| `.txt`  | **Um valor por linha**, sem separadores adicionais |\n"
    "| `.csv`  | Primeira coluna utilizada; cabeçalho não-numérico ignorado |\n"
    "| `.xlsx` | Primeira coluna da primeira planilha |\n\n"
    "**Exemplo (.txt / .csv):**\n```\n1.23\n4.56\n7.89\n```"
)


# ─────────────────────────────────────────────────────────────────────────────
# CABEÇALHO E ABAS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(HEADER_HTML, unsafe_allow_html=True)

tab_config, tab_results, tab_charts = st.tabs(
    ["⚙️  Configuração", "📊  Resultados", "📈  Gráficos"]
)


# ─────────────────────────────────────────────────────────────────────────────
# ABA CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

with tab_config:
    st.markdown("### 📁 Arquivos de Entrada")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="upload-hint">🔷 Xcal</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-desc">Calibração - X (matriz) — .txt · .csv · .xlsx</div>', unsafe_allow_html=True)
        xcal_file = st.file_uploader(
            "Xcal — matriz", type=_TIPOS_MATRIZ, key="xcal",
            label_visibility="collapsed", help=_HELP_MATRIZ,
        )
        if xcal_file:
            st.success(f"✅ Xcal carregado: {xcal_file.name}")
        with st.expander("ℹ️ Como formatar o Xcal"):
            st.markdown(_EXP_MATRIZ)

    with col2:
        st.markdown('<div class="upload-hint">📉 Ycal</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-desc">Calibração - Y (vetor) — .txt · .csv · .xlsx</div>', unsafe_allow_html=True)
        ycal_file = st.file_uploader(
            "Ycal — vetor", type=_TIPOS_VETOR, key="ycal",
            label_visibility="collapsed", help=_HELP_VETOR,
        )
        if ycal_file:
            st.success(f"✅ Ycal carregado: {ycal_file.name}")
        with st.expander("ℹ️ Como formatar o Ycal"):
            st.markdown(_EXP_VETOR)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown('<div class="upload-hint">🔹 Xteste</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-desc">Teste - X (matriz) — .txt · .csv · .xlsx</div>', unsafe_allow_html=True)
        xteste_file = st.file_uploader(
            "Xteste — matriz", type=_TIPOS_MATRIZ, key="xteste",
            label_visibility="collapsed", help=_HELP_MATRIZ,
        )
        if xteste_file:
            st.success(f"✅ Xteste carregado: {xteste_file.name}")
        with st.expander("ℹ️ Como formatar o Xteste"):
            st.markdown(_EXP_MATRIZ)

    with col4:
        st.markdown('<div class="upload-hint">📈 Yteste</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-desc">Teste - Y (vetor) — .txt · .csv · .xlsx</div>', unsafe_allow_html=True)
        yteste_file = st.file_uploader(
            "Yteste — vetor", type=_TIPOS_VETOR, key="yteste",
            label_visibility="collapsed", help=_HELP_VETOR,
        )
        if yteste_file:
            st.success(f"✅ Yteste carregado: {yteste_file.name}")
        with st.expander("ℹ️ Como formatar o Yteste"):
            st.markdown(_EXP_VETOR)

    st.markdown("---")
    st.markdown("### ⚙️ Parâmetros de Otimização")

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        m_input = st.text_input(
            "Valores de m (bags)",
            value="10, 20, 30, 40, 50",
            help="Inteiros positivos separados por vírgula. Ex: 10, 20, 30, 40, 50",
        )
    with pcol2:
        k_input = st.text_input(
            "Valores de k (subagging)",
            value="0.10, 0.20, 0.30, 0.40, 0.50",
            help="Decimais entre 0 e 1 separados por vírgula. Ex: 0.5, 0.6, 0.7, 0.8",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    files_ok = all([xcal_file, ycal_file, xteste_file, yteste_file])
    if not files_ok:
        st.info("⚠️ Selecione os 4 arquivos (Xcal, Ycal, Xteste, Yteste) para habilitar a análise.")

    if st.button("🚀  Executar Análise com Otimização CV", disabled=not files_ok):

        # Validação dos parâmetros
        try:
            m_values = parse_int_list(m_input)
            k_values = parse_float_list(k_input)
        except ValueError as e:
            st.error(f"❌ {e}")
            st.stop()

        # Leitura dos arquivos — parse_matrix/parse_vector recebem o file_obj diretamente
        try:
            Xcal   = parse_matrix(xcal_file)
            Ycal   = parse_vector(ycal_file)
            Xteste = parse_matrix(xteste_file)
            Yteste = parse_vector(yteste_file)
        except Exception as e:
            st.error(f"❌ Erro ao ler arquivos: {e}")
            st.stop()

        # Barra de progresso
        progress_bar = st.progress(0, text="⏳ Processando análise com otimização por validação cruzada...")

        def update_progress(val: float) -> None:
            progress_bar.progress(min(val * 0.95, 0.95), text="⏳ Otimizando parâmetros...")

        # Execução — delega inteiramente a models.py
        try:
            result = run_full_analysis(Xcal, Ycal, Xteste, Yteste, m_values, k_values, update_progress)
            progress_bar.progress(1.0, text="✅ Concluído!")
            st.session_state["result"] = result
            st.success("✅ Análise concluída com sucesso! Veja as abas **Resultados** e **Gráficos**.")
        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ Erro na análise: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ABA RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

with tab_results:
    if "result" not in st.session_state:
        st.info("Os resultados serão exibidos após a execução da análise.")
    else:
        result  = st.session_state["result"]
        metrics = result["metrics"]
        optimal = result["optimal"]

        st.markdown("### 📊 Comparação de Métricas")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                metric_card_html("RLM Simples", "─", COR_RLM, metrics["rlm"]),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                metric_card_html(
                    f"RLM + Bagging (m*={optimal['m_bagging']})",
                    "🅱", COR_BAGGING, metrics["bagging"],
                ),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                metric_card_html(
                    f"RLM + Subagging (k*={optimal['k_subagging']*100:.0f}%, m*={optimal['m_subagging']})",
                    "🆂", COR_SUBAGGING, metrics["subagging"],
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Tabela Comparativa")

        df_metrics = pd.DataFrame({
            "Técnica":    ["RLM Simples", "RLM + Bagging", "RLM + Subagging"],
            "RMSE":       [metrics["rlm"]["rmse"],     metrics["bagging"]["rmse"],  metrics["subagging"]["rmse"]],
            "MAE":        [metrics["rlm"]["mae"],      metrics["bagging"]["mae"],   metrics["subagging"]["mae"]],
            "R²":         [metrics["rlm"]["r2"],       metrics["bagging"]["r2"],    metrics["subagging"]["r2"]],
            "Tempo (ms)": [metrics["rlm"]["time"],     metrics["bagging"]["time"],  metrics["subagging"]["time"]],
        })
        st.dataframe(
            df_metrics.style.format({
                "RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}", "Tempo (ms)": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        csv = df_metrics.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Baixar métricas como CSV", csv, "metricas.csv", "text/csv")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🖥️ Log de Execução")
        st.code(result["log"], language=None)
        st.download_button(
            "⬇️ Baixar log completo",
            result["log"].encode("utf-8"),
            "relatorio.txt",
            "text/plain",
        )


# ─────────────────────────────────────────────────────────────────────────────
# ABA GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────

with tab_charts:
    if "result" not in st.session_state:
        st.info("Os gráficos serão exibidos após a execução da análise.")
    else:
        result  = st.session_state["result"]
        preds   = result["preds"]
        optimal = result["optimal"]

        st.markdown("### 📈 Valores Reais vs Preditos")

        # Gráfico 1 — RLM Simples
        st.plotly_chart(
            build_chart(preds["y_real"], preds["y_pred_rlm"], "RLM Simples", COR_RLM),
            use_container_width=True,
        )

        # Gráfico 2 — Bagging
        st.plotly_chart(
            build_chart(
                preds["y_real"], preds["y_pred_bag"],
                f"RLM + Bagging (m*={optimal['m_bagging']})",
                COR_BAGGING,
            ),
            use_container_width=True,
        )

        # Gráfico 3 — Subagging
        st.plotly_chart(
            build_chart(
                preds["y_real"], preds["y_pred_sub"],
                f"RLM + Subagging (k*={optimal['k_subagging']*100:.0f}%, m*={optimal['m_subagging']})",
                COR_SUBAGGING,
            ),
            use_container_width=True,
        )

        # Gráfico 4 — Dispersão Real × Predito (todos os modelos)
        st.markdown("### 🎯 Dispersão Real × Predito")
        st.plotly_chart(build_scatter_chart(preds), use_container_width=True)
