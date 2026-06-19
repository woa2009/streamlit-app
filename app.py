"""
app.py — Orquestrador da Interface
Regressão Espectral: RLM, Bagging e Subagging

Responsabilidade exclusiva: layout Streamlit, abas e interação com o usuário.
Toda a matemática vem de models.py.
Todos os gráficos e componentes visuais vêm de components.py.

Autor: Wagner Oliveira de Araujo
Versão: 8
"""

import streamlit as st
import pandas as pd

from models import parse_matrix, parse_vector, run_full_analysis
from components import (
    APP_CSS,
    COR_RLM,
    COR_BAGGING,
    COR_SUBAGGING,
    get_header_html,
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
# DICIONÁRIO DE TRADUÇÕES
# ─────────────────────────────────────────────────────────────────────────────

_TRADUCOES = {
    "pt": {
        # Seletor de idioma
        "lang_label":               "🌐 Idioma / Language",
        # Abas
        "tab_config":               "⚙️  Configuração",
        "tab_results":              "📊  Resultados",
        "tab_charts":               "📈  Gráficos",
        # Aba Configuração — seção de arquivos
        "sec_arquivos":             "### 📁 Arquivos de Entrada",
        "hint_xcal":                "🔷 Xcal",
        "desc_xcal":                "Calibração - X (matriz) — .txt · .csv · .xlsx",
        "exp_xcal":                 "ℹ️ Como formatar o Xcal",
        "success_xcal":             "✅ Xcal carregado: ",
        "hint_ycal":                "📉 Ycal",
        "desc_ycal":                "Calibração - Y (vetor) — .txt · .csv · .xlsx",
        "exp_ycal":                 "ℹ️ Como formatar o Ycal",
        "success_ycal":             "✅ Ycal carregado: ",
        "hint_xteste":              "🔹 Xteste",
        "desc_xteste":              "Teste - X (matriz) — .txt · .csv · .xlsx",
        "exp_xteste":               "ℹ️ Como formatar o Xteste",
        "success_xteste":           "✅ Xteste carregado: ",
        "hint_yteste":              "📈 Yteste",
        "desc_yteste":              "Teste - Y (vetor) — .txt · .csv · .xlsx",
        "exp_yteste":               "ℹ️ Como formatar o Yteste",
        "success_yteste":           "✅ Yteste carregado: ",
        # Aba Configuração — parâmetros
        "sec_params":               "### ⚙️ Parâmetros de Otimização",
        "label_m":                  "Valores de m (bags)",
        "help_m":                   "Inteiros positivos separados por vírgula. Ex: 10, 20, 30, 40, 50",
        "label_k":                  "Valores de k (subagging)",
        "help_k":                   "Decimais entre 0 e 1 separados por vírgula. Ex: 0.5, 0.6, 0.7, 0.8",
        "info_arquivos":            "⚠️ Selecione os 4 arquivos (Xcal, Ycal, Xteste, Yteste) para habilitar a análise.",
        "btn_executar":             "🚀  Executar Análise com Otimização CV",
        # Progresso e mensagens de execução
        "prog_inicio":              "⏳ Processando análise com otimização por validação cruzada...",
        "prog_otimizando":          "⏳ Otimizando parâmetros...",
        "prog_concluido":           "✅ Concluído!",
        "success_analise":          "✅ Análise concluída com sucesso! Veja as abas **Resultados** e **Gráficos**.",
        "erro_params":              "❌ ",
        "erro_arquivos":            "❌ Erro ao ler arquivos: ",
        "erro_analise":             "❌ Erro na análise: ",
        # Aba Resultados
        "sec_metricas":             "### 📊 Comparação de Métricas",
        "info_sem_resultado":       "Os resultados serão exibidos após a execução da análise.",
        "sec_tabela":               "### 📋 Tabela Comparativa",
        "col_tecnica":              "Técnica",
        "col_tempo":                "Tempo (ms)",
        "nome_rlm":                 "RLM Simples",
        "nome_bagging":             "RLM + Bagging",
        "nome_subagging":           "RLM + Subagging",
        "btn_csv":                  "⬇️ Baixar métricas como CSV",
        "csv_filename":             "metricas.csv",
        "sec_log":                  "### 🖥️ Log de Execução",
        "btn_log":                  "⬇️ Baixar log completo",
        "log_filename":             "relatorio.txt",
        # Aba Gráficos
        "info_sem_graficos":        "Os gráficos serão exibidos após a execução da análise.",
        "sec_graficos":             "### 📈 Valores Reais vs Preditos",
        "sec_dispersao":            "### 🎯 Dispersão Real × Predito",
        # Textos de ajuda — matriz
        "help_matriz": (
            "📐 Formato de matriz — aceita .txt · .csv · .xlsx\n\n"
            "• .txt  → colunas separadas por espaços simples, uma amostra por linha.\n"
            "• .csv  → separador de coluna detectado automaticamente (vírgula ou ponto-e-vírgula).\n"
            "• .xlsx → primeira planilha, sem linha de cabeçalho; cada linha = uma amostra.\n\n"
            "Não misture separadores de coluna e decimal no mesmo arquivo."
        ),
        "exp_matriz": (
            "**Formatos aceitos:** `.txt` · `.csv` · `.xlsx`\n\n"
            "| Formato | Regra |\n"
            "|---------|-------|\n"
            "| `.txt`  | Colunas separadas por **espaços simples**, uma amostra por linha |\n"
            "| `.csv`  | Separador de coluna: **vírgula** ou **ponto-e-vírgula** (detectado automaticamente) |\n"
            "| `.xlsx` | Primeira planilha, sem linha de cabeçalho |\n\n"
            "**Exemplo (.txt):**\n```\n0.123 0.456 0.789\n0.321 0.654 0.987\n0.111 0.222 0.333\n```\n\n"
            "**Exemplo (.csv com `;`):**\n```\n0.123;0.456;0.789\n0.321;0.654;0.987\n```"
        ),
        # Textos de ajuda — vetor
        "help_vetor": (
            "📋 Formato de vetor coluna — aceita .txt · .csv · .xlsx\n\n"
            "• .txt  → um valor por linha, sem separadores adicionais.\n"
            "• .csv  → primeira coluna utilizada; cabeçalho não-numérico ignorado.\n"
            "• .xlsx → primeira coluna da primeira planilha.\n\n"
            "Não utilize vírgulas ou ponto e vírgula em arquivos .txt."
        ),
        "exp_vetor": (
            "**Formatos aceitos:** `.txt` · `.csv` · `.xlsx`\n\n"
            "| Formato | Regra |\n"
            "|---------|-------|\n"
            "| `.txt`  | **Um valor por linha**, sem separadores adicionais |\n"
            "| `.csv`  | Primeira coluna utilizada; cabeçalho não-numérico ignorado |\n"
            "| `.xlsx` | Primeira coluna da primeira planilha |\n\n"
            "**Exemplo (.txt / .csv):**\n```\n1.23\n4.56\n7.89\n```"
        ),
    },

    "en": {
        # Seletor de idioma
        "lang_label":               "🌐 Language / Idioma",
        # Abas
        "tab_config":               "⚙️  Configuration",
        "tab_results":              "📊  Results",
        "tab_charts":               "📈  Charts",
        # Aba Configuração — seção de arquivos
        "sec_arquivos":             "### 📁 Input Files",
        "hint_xcal":                "🔷 Xcal",
        "desc_xcal":                "Calibration - X (matrix) — .txt · .csv · .xlsx",
        "exp_xcal":                 "ℹ️ How to format Xcal",
        "success_xcal":             "✅ Xcal loaded: ",
        "hint_ycal":                "📉 Ycal",
        "desc_ycal":                "Calibration - Y (vector) — .txt · .csv · .xlsx",
        "exp_ycal":                 "ℹ️ How to format Ycal",
        "success_ycal":             "✅ Ycal loaded: ",
        "hint_xteste":              "🔹 Xtest",
        "desc_xteste":              "Test - X (matrix) — .txt · .csv · .xlsx",
        "exp_xteste":               "ℹ️ How to format Xtest",
        "success_xteste":           "✅ Xtest loaded: ",
        "hint_yteste":              "📈 Ytest",
        "desc_yteste":              "Test - Y (vector) — .txt · .csv · .xlsx",
        "exp_yteste":               "ℹ️ How to format Ytest",
        "success_yteste":           "✅ Ytest loaded: ",
        # Aba Configuração — parâmetros
        "sec_params":               "### ⚙️ Optimization Parameters",
        "label_m":                  "m values (bags)",
        "help_m":                   "Positive integers separated by comma. Ex: 10, 20, 30, 40, 50",
        "label_k":                  "k values (subagging)",
        "help_k":                   "Decimals between 0 and 1 separated by comma. Ex: 0.5, 0.6, 0.7, 0.8",
        "info_arquivos":            "⚠️ Select the 4 files (Xcal, Ycal, Xtest, Ytest) to enable the analysis.",
        "btn_executar":             "🚀  Run Analysis with CV Optimization",
        # Progresso e mensagens de execução
        "prog_inicio":              "⏳ Processing analysis with cross-validation optimization...",
        "prog_otimizando":          "⏳ Optimizing parameters...",
        "prog_concluido":           "✅ Done!",
        "success_analise":          "✅ Analysis completed successfully! See the **Results** and **Charts** tabs.",
        "erro_params":              "❌ ",
        "erro_arquivos":            "❌ Error reading files: ",
        "erro_analise":             "❌ Error during analysis: ",
        # Aba Resultados
        "sec_metricas":             "### 📊 Metrics Comparison",
        "info_sem_resultado":       "Results will be displayed after running the analysis.",
        "sec_tabela":               "### 📋 Comparison Table",
        "col_tecnica":              "Method",
        "col_tempo":                "Time (ms)",
        "nome_rlm":                 "Simple MLR",
        "nome_bagging":             "MLR + Bagging",
        "nome_subagging":           "MLR + Subagging",
        "btn_csv":                  "⬇️ Download metrics as CSV",
        "csv_filename":             "metrics.csv",
        "sec_log":                  "### 🖥️ Execution Log",
        "btn_log":                  "⬇️ Download full log",
        "log_filename":             "report.txt",
        # Aba Gráficos
        "info_sem_graficos":        "Charts will be displayed after running the analysis.",
        "sec_graficos":             "### 📈 Actual vs Predicted Values",
        "sec_dispersao":            "### 🎯 Actual × Predicted Scatter",
        # Textos de ajuda — matriz
        "help_matriz": (
            "📐 Matrix format — accepts .txt · .csv · .xlsx\n\n"
            "• .txt  → columns separated by single spaces, one sample per row.\n"
            "• .csv  → column separator auto-detected (comma or semicolon).\n"
            "• .xlsx → first sheet, no header row; each row = one sample.\n\n"
            "Do not mix column and decimal separators in the same file."
        ),
        "exp_matriz": (
            "**Accepted formats:** `.txt` · `.csv` · `.xlsx`\n\n"
            "| Format | Rule |\n"
            "|--------|------|\n"
            "| `.txt`  | Columns separated by **single spaces**, one sample per row |\n"
            "| `.csv`  | Column separator: **comma** or **semicolon** (auto-detected) |\n"
            "| `.xlsx` | First sheet, no header row |\n\n"
            "**Example (.txt):**\n```\n0.123 0.456 0.789\n0.321 0.654 0.987\n0.111 0.222 0.333\n```\n\n"
            "**Example (.csv with `;`):**\n```\n0.123;0.456;0.789\n0.321;0.654;0.987\n```"
        ),
        # Textos de ajuda — vetor
        "help_vetor": (
            "📋 Column vector format — accepts .txt · .csv · .xlsx\n\n"
            "• .txt  → one value per row, no additional separators.\n"
            "• .csv  → first column used; non-numeric header ignored.\n"
            "• .xlsx → first column of the first sheet.\n\n"
            "Do not use commas or semicolons in .txt files."
        ),
        "exp_vetor": (
            "**Accepted formats:** `.txt` · `.csv` · `.xlsx`\n\n"
            "| Format | Rule |\n"
            "|--------|------|\n"
            "| `.txt`  | **One value per row**, no additional separators |\n"
            "| `.csv`  | First column used; non-numeric header ignored |\n"
            "| `.xlsx` | First column of the first sheet |\n\n"
            "**Example (.txt / .csv):**\n```\n1.23\n4.56\n7.89\n```"
        ),
    },
}


def _t(chave: str) -> str:
    """Retorna a tradução de uma chave para o idioma ativo em st.session_state."""
    lang = st.session_state.get("lang", "pt")
    return _TRADUCOES[lang][chave]


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
# SELETOR DE IDIOMA (sidebar)
# ─────────────────────────────────────────────────────────────────────────────

if "lang" not in st.session_state:
    st.session_state["lang"] = "pt"

with st.sidebar:
    idioma_opcoes = {"Português": "pt", "English": "en"}
    idioma_selecionado = st.radio(
        _t("lang_label"),
        options=list(idioma_opcoes.keys()),
        index=0 if st.session_state["lang"] == "pt" else 1,
        key="lang_radio",
    )
    st.session_state["lang"] = idioma_opcoes[idioma_selecionado]

lang = st.session_state["lang"]


# ─────────────────────────────────────────────────────────────────────────────
# CABEÇALHO E ABAS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(get_header_html(lang), unsafe_allow_html=True)

tab_config, tab_results, tab_charts = st.tabs([
    _t("tab_config"),
    _t("tab_results"),
    _t("tab_charts"),
])

_TIPOS_MATRIZ = ["txt", "csv", "xlsx"]
_TIPOS_VETOR  = ["txt", "csv", "xlsx"]


# ─────────────────────────────────────────────────────────────────────────────
# ABA CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

with tab_config:
    st.markdown(_t("sec_arquivos"))

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f'<div class="upload-hint">{_t("hint_xcal")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="upload-desc">{_t("desc_xcal")}</div>', unsafe_allow_html=True)
        xcal_file = st.file_uploader(
            "Xcal", type=_TIPOS_MATRIZ, key="xcal",
            label_visibility="collapsed", help=_t("help_matriz"),
        )
        if xcal_file:
            st.success(_t("success_xcal") + xcal_file.name)
        with st.expander(_t("exp_xcal")):
            st.markdown(_t("exp_matriz"))

    with col2:
        st.markdown(f'<div class="upload-hint">{_t("hint_ycal")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="upload-desc">{_t("desc_ycal")}</div>', unsafe_allow_html=True)
        ycal_file = st.file_uploader(
            "Ycal", type=_TIPOS_VETOR, key="ycal",
            label_visibility="collapsed", help=_t("help_vetor"),
        )
        if ycal_file:
            st.success(_t("success_ycal") + ycal_file.name)
        with st.expander(_t("exp_ycal")):
            st.markdown(_t("exp_vetor"))

    col3, col4 = st.columns(2)
    with col3:
        st.markdown(f'<div class="upload-hint">{_t("hint_xteste")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="upload-desc">{_t("desc_xteste")}</div>', unsafe_allow_html=True)
        xteste_file = st.file_uploader(
            "Xteste", type=_TIPOS_MATRIZ, key="xteste",
            label_visibility="collapsed", help=_t("help_matriz"),
        )
        if xteste_file:
            st.success(_t("success_xteste") + xteste_file.name)
        with st.expander(_t("exp_xteste")):
            st.markdown(_t("exp_matriz"))

    with col4:
        st.markdown(f'<div class="upload-hint">{_t("hint_yteste")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="upload-desc">{_t("desc_yteste")}</div>', unsafe_allow_html=True)
        yteste_file = st.file_uploader(
            "Yteste", type=_TIPOS_VETOR, key="yteste",
            label_visibility="collapsed", help=_t("help_vetor"),
        )
        if yteste_file:
            st.success(_t("success_yteste") + yteste_file.name)
        with st.expander(_t("exp_yteste")):
            st.markdown(_t("exp_vetor"))

    st.markdown("---")
    st.markdown(_t("sec_params"))

    pcol1, pcol2 = st.columns(2)
    with pcol1:
        m_input = st.text_input(
            _t("label_m"),
            value="10, 20, 30, 40, 50",
            help=_t("help_m"),
        )
    with pcol2:
        k_input = st.text_input(
            _t("label_k"),
            value="0.10, 0.20, 0.30, 0.40, 0.50",
            help=_t("help_k"),
        )

    st.markdown("<br>", unsafe_allow_html=True)

    files_ok = all([xcal_file, ycal_file, xteste_file, yteste_file])
    if not files_ok:
        st.info(_t("info_arquivos"))

    if st.button(_t("btn_executar"), disabled=not files_ok):

        # Validação dos parâmetros
        try:
            m_values = parse_int_list(m_input)
            k_values = parse_float_list(k_input)
        except ValueError as e:
            st.error(_t("erro_params") + str(e))
            st.stop()

        # Leitura dos arquivos — parse_matrix/parse_vector recebem o file_obj diretamente
        try:
            Xcal   = parse_matrix(xcal_file)
            Ycal   = parse_vector(ycal_file)
            Xteste = parse_matrix(xteste_file)
            Yteste = parse_vector(yteste_file)
        except Exception as e:
            st.error(_t("erro_arquivos") + str(e))
            st.stop()

        # Barra de progresso
        progress_bar = st.progress(0, text=_t("prog_inicio"))

        def update_progress(val: float) -> None:
            progress_bar.progress(min(val * 0.95, 0.95), text=_t("prog_otimizando"))

        # Execução — delega inteiramente a models.py
        try:
            result = run_full_analysis(Xcal, Ycal, Xteste, Yteste, m_values, k_values, update_progress, lang)
            progress_bar.progress(1.0, text=_t("prog_concluido"))
            st.session_state["result"] = result
            st.success(_t("success_analise"))
        except Exception as e:
            progress_bar.empty()
            st.error(_t("erro_analise") + str(e))


# ─────────────────────────────────────────────────────────────────────────────
# ABA RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

with tab_results:
    if "result" not in st.session_state:
        st.info(_t("info_sem_resultado"))
    else:
        result  = st.session_state["result"]
        metrics = result["metrics"]
        optimal = result["optimal"]

        st.markdown(_t("sec_metricas"))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                metric_card_html(_t("nome_rlm"), "─", COR_RLM, metrics["rlm"], lang),
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                metric_card_html(
                    f"{_t('nome_bagging')} (m*={optimal['m_bagging']})",
                    "🅱", COR_BAGGING, metrics["bagging"], lang,
                ),
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                metric_card_html(
                    f"{_t('nome_subagging')} (k*={optimal['k_subagging']*100:.0f}%, m*={optimal['m_subagging']})",
                    "🆂", COR_SUBAGGING, metrics["subagging"], lang,
                ),
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(_t("sec_tabela"))

        df_metrics = pd.DataFrame({
            _t("col_tecnica"): [_t("nome_rlm"), _t("nome_bagging"), _t("nome_subagging")],
            "RMSE":            [metrics["rlm"]["rmse"],  metrics["bagging"]["rmse"],  metrics["subagging"]["rmse"]],
            "MAE":             [metrics["rlm"]["mae"],   metrics["bagging"]["mae"],   metrics["subagging"]["mae"]],
            "R²":              [metrics["rlm"]["r2"],    metrics["bagging"]["r2"],    metrics["subagging"]["r2"]],
            _t("col_tempo"):   [metrics["rlm"]["time"],  metrics["bagging"]["time"],  metrics["subagging"]["time"]],
        })
        st.dataframe(
            df_metrics.style.format({
                "RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}",
                _t("col_tempo"): "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        csv = df_metrics.to_csv(index=False).encode("utf-8")
        st.download_button(_t("btn_csv"), csv, _t("csv_filename"), "text/csv")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(_t("sec_log"))
        st.code(result["log"], language=None)
        st.download_button(
            _t("btn_log"),
            result["log"].encode("utf-8"),
            _t("log_filename"),
            "text/plain",
        )


# ─────────────────────────────────────────────────────────────────────────────
# ABA GRÁFICOS
# ─────────────────────────────────────────────────────────────────────────────

with tab_charts:
    if "result" not in st.session_state:
        st.info(_t("info_sem_graficos"))
    else:
        result  = st.session_state["result"]
        preds   = result["preds"]
        optimal = result["optimal"]

        st.markdown(_t("sec_graficos"))

        # Gráfico 1 — RLM Simples
        st.plotly_chart(
            build_chart(preds["y_real"], preds["y_pred_rlm"], _t("nome_rlm"), COR_RLM, lang),
            use_container_width=True,
        )

        # Gráfico 2 — Bagging
        st.plotly_chart(
            build_chart(
                preds["y_real"], preds["y_pred_bag"],
                f"{_t('nome_bagging')} (m*={optimal['m_bagging']})",
                COR_BAGGING, lang,
            ),
            use_container_width=True,
        )

        # Gráfico 3 — Subagging
        st.plotly_chart(
            build_chart(
                preds["y_real"], preds["y_pred_sub"],
                f"{_t('nome_subagging')} (k*={optimal['k_subagging']*100:.0f}%, m*={optimal['m_subagging']})",
                COR_SUBAGGING, lang,
            ),
            use_container_width=True,
        )

        # Gráfico 4 — Dispersão Real × Predito (todos os modelos)
        st.markdown(_t("sec_dispersao"))
        st.plotly_chart(build_scatter_chart(preds, lang), use_container_width=True)
