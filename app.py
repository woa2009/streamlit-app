"""
Regressão Espectral - Bagging & Subagging
Conversão do projeto Vaadin Flow (Java) para Streamlit (Python)

Autor original: Wagner Oliveira de Araujo
Versão Streamlit: Conversão automática
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import io

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Regressão Espectral - Bagging & Subagging",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS personalizado (replica o visual do Vaadin)
st.markdown("""
<style>
    /* Cabeçalho */
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8em; }
    .main-header p  { color: rgba(255,255,255,0.85); margin: 4px 0 0 0; font-size: 0.9em; }

    /* Cards de métricas */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.1);
        height: 100%;
    }
    .metric-card-rlm      { border-left: 5px solid #1565c0; }
    .metric-card-bagging  { border-left: 5px solid #2e7d32; }
    .metric-card-subagging{ border-left: 5px solid #6a1b9a; }

    /* Log */
    .log-area {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 16px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.82em;
        white-space: pre-wrap;
        overflow-y: auto;
        max-height: 500px;
        line-height: 1.5;
    }

    /* Upload cards */
    .upload-hint {
        color: #1565c0;
        font-weight: 700;
        font-size: 1.05em;
        margin-bottom: 4px;
    }
    .upload-desc {
        color: #666;
        font-size: 0.85em;
        margin-bottom: 8px;
    }

    /* Status badges */
    .status-ok  { color: #2e7d32; font-weight: 600; }
    .status-err { color: #c62828; font-weight: 600; }

    /* Botão executar */
    div.stButton > button {
        background: linear-gradient(135deg, #2e7d32 0%, #43a047 100%);
        color: white;
        font-size: 1.1em;
        height: 52px;
        width: 100%;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LÓGICA DE REGRESSÃO  (tradução direta do RegressionService.java)
# ─────────────────────────────────────────────────────────────────────────────

def _file_ext(file_obj) -> str:
    """Retorna extensão do arquivo em minúsculas (ex.: 'csv', 'xlsx', 'txt')."""
    name = getattr(file_obj, "name", "")
    return name.rsplit(".", 1)[-1].lower() if "." in name else "txt"


def _detect_csv_sep(text: str) -> str:
    """Detecta separador de colunas do CSV: ';' tem prioridade sobre ','."""
    return ";" if text.count(";") >= text.count(",") else ","


def parse_matrix(file_obj) -> np.ndarray:
    """
    Lê uma matriz numérica de um arquivo .txt, .csv ou .xlsx.

    .txt  → colunas separadas por espaço/tab; decimal pode ser vírgula (BR) ou ponto.
    .csv  → separador de coluna detectado automaticamente (vírgula ou ponto-e-vírgula);
            se o separador for ';', vírgulas são tratadas como decimal (padrão BR).
    .xlsx → primeira planilha; sem linha de cabeçalho; cada linha = uma amostra.
    """
    ext = _file_ext(file_obj)
    raw = file_obj.read()

    if ext == "xlsx":
        df = pd.read_excel(io.BytesIO(raw), header=None, engine="openpyxl")
        return df.apply(pd.to_numeric, errors="coerce").dropna(how="all").values.astype(float)

    text = raw.decode("utf-8", errors="ignore")

    if ext == "csv":
        sep = _detect_csv_sep(text)
        if sep == ";":                        # decimal BR: troca ',' por '.'
            text = text.replace(",", ".")
        df = pd.read_csv(io.StringIO(text), sep=sep, header=None)
        return df.apply(pd.to_numeric, errors="coerce").dropna(how="all").values.astype(float)

    # .txt (comportamento original, suporta decimal BR)
    rows = []
    for line in text.splitlines():
        line = line.strip().replace(",", ".")
        if not line:
            continue
        rows.append([float(v) for v in line.split()])
    return np.array(rows)


def parse_vector(file_obj) -> np.ndarray:
    """
    Lê um vetor numérico (coluna) de um arquivo .txt, .csv ou .xlsx.

    .txt  → um valor por linha.
    .csv  → primeira coluna utilizada; cabeçalho não-numérico ignorado automaticamente.
    .xlsx → primeira coluna da primeira planilha.
    """
    ext = _file_ext(file_obj)
    raw = file_obj.read()

    if ext == "xlsx":
        df = pd.read_excel(io.BytesIO(raw), header=None, engine="openpyxl")
        col = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        return col.values.astype(float)

    text = raw.decode("utf-8", errors="ignore")

    if ext == "csv":
        sep = _detect_csv_sep(text)
        if sep == ";":
            text = text.replace(",", ".")
        df = pd.read_csv(io.StringIO(text), sep=sep, header=None)
        col = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna()
        return col.values.astype(float)

    # .txt
    data = []
    for line in text.splitlines():
        line = line.strip().replace(",", ".")
        if not line:
            continue
        data.append(float(line))
    return np.array(data)


def perform_rlm(Xcal, Ycal, Xteste):
    """RLM simples via pseudoinversa."""
    beta = np.linalg.pinv(Xcal) @ Ycal
    return Xteste @ beta


def perform_bagging(Xcal, Ycal, Xteste, num_bags, with_replacement, fraction, seed=42):
    """RLM + Bagging / Subagging."""
    n = Xcal.shape[0]
    sample_size = n if with_replacement else max(1, int(n * fraction))
    rng = np.random.default_rng(seed)
    Ypred = np.zeros(Xteste.shape[0])

    for _ in range(num_bags):
        if with_replacement:
            idx = rng.integers(0, n, size=sample_size)
        else:
            idx = rng.choice(n, size=sample_size, replace=False)

        Xbag = Xcal[idx]
        Ybag = Ycal[idx]
        beta = np.linalg.pinv(Xbag) @ Ybag
        Ypred += (Xteste @ beta) / num_bags

    return Ypred


def calc_metrics(actual, predicted):
    """Calcula RMSE, MAE e R²."""
    err  = predicted - actual
    rmse = np.sqrt(np.mean(err ** 2))
    mae  = np.mean(np.abs(err))
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    ss_res = np.sum(err ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    return rmse, mae, r2


def run_cross_validation(Xcal, Ycal, m, with_replacement, fraction, n_folds=5, seed=42):
    """Validação cruzada k-fold. Retorna (mean_rmse, std_rmse)."""
    n = Xcal.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    rmse_list = []

    for fold in range(n_folds):
        fold_size = n // n_folds
        start = fold * fold_size
        end   = n if fold == n_folds - 1 else (fold + 1) * fold_size

        val_idx   = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        Xtrain, Ytrain = Xcal[train_idx], Ycal[train_idx]
        Xval,   Yval   = Xcal[val_idx],   Ycal[val_idx]

        Ypred = perform_bagging(Xtrain, Ytrain, Xval, m, with_replacement, fraction, seed)
        rmse_list.append(np.sqrt(np.mean((Ypred - Yval) ** 2)))

    return float(np.mean(rmse_list)), float(np.std(rmse_list))


def find_optimal_parameters(Xcal, Ycal, m_values, k_values, progress_callback=None):
    """Otimiza hiperparâmetros por CV 5-fold (espelho do Java)."""
    log = []
    log.append("════════════════════════════════════════════════════════════════")
    log.append("     ETAPA 2: OTIMIZAÇÃO DE PARÂMETROS (Validação Cruzada 5-Fold)")
    log.append("════════════════════════════════════════════════════════════════\n")
    log.append(f"📋 Valores de m testados: {m_values}")
    log.append(f"📋 Valores de k testados: {k_values}\n")

    total_steps = len(m_values) + len(m_values) * len(k_values)
    step = 0

    # Otimiza Bagging
    log.append("🔍 OTIMIZANDO RLM-BAGGING (testando m):")
    log.append("─────────────────────────────────────────")
    best_bag_rmse, best_bag_m = float("inf"), m_values[0]
    best_bag_rmse_cv = 0.0

    for m in m_values:
        mean_r, std_r = run_cross_validation(Xcal, Ycal, m, True, 1.0)
        log.append(f"   m={m:3d} → RMSE-CV = {mean_r:.4f} ± {std_r:.4f}")
        if mean_r < best_bag_rmse:
            best_bag_rmse, best_bag_m, best_bag_rmse_cv = mean_r, m, mean_r
        step += 1
        if progress_callback:
            progress_callback(step / total_steps)

    log.append(f"\n✅ Melhor Bagging: m* = {best_bag_m} (RMSE-CV = {best_bag_rmse:.4f})\n")

    # Otimiza Subagging
    log.append("🔍 OTIMIZANDO RLM-SUBAGGING (testando k × m):")
    log.append("─────────────────────────────────────────────────")
    best_sub_rmse, best_sub_m, best_sub_k = float("inf"), m_values[0], k_values[0]
    best_sub_rmse_cv = 0.0

    for k in k_values:
        log.append(f"\n   Testando k={k*100:.0f}%:")
        for m in m_values:
            mean_r, std_r = run_cross_validation(Xcal, Ycal, m, False, k)
            log.append(f"      m={m:3d} → RMSE-CV = {mean_r:.4f} ± {std_r:.4f}")
            if mean_r < best_sub_rmse:
                best_sub_rmse, best_sub_m, best_sub_k, best_sub_rmse_cv = mean_r, m, k, mean_r
            step += 1
            if progress_callback:
                progress_callback(step / total_steps)

    log.append(f"\n✅ Melhor Subagging: k* = {best_sub_k*100:.0f}%, m* = {best_sub_m} (RMSE-CV = {best_sub_rmse:.4f})\n")

    optimal = {
        "m_bagging":        best_bag_m,
        "m_subagging":      best_sub_m,
        "k_subagging":      best_sub_k,
        "rmse_cv_bagging":  best_bag_rmse_cv,
        "rmse_cv_subagging":best_sub_rmse_cv,
    }
    return optimal, "\n".join(log)


def run_full_analysis(Xcal, Ycal, Xteste, Yteste, m_values, k_values, progress_cb=None):
    """Executa análise completa — espelho do RegressionService.runAnalysis()."""
    log_parts = []
    log_parts.append("════════════════════════════════════════════════════════════════")
    log_parts.append("   ANÁLISE COM OTIMIZAÇÃO POR VALIDAÇÃO CRUZADA 5-FOLD")
    log_parts.append("════════════════════════════════════════════════════════════════\n")
    log_parts.append(f"📋 Valores de m (bags): {m_values}")
    log_parts.append(f"📋 Valores de k (subagging): {k_values}\n")
    log_parts.append("=== DIMENSÕES DETECTADAS ===")
    log_parts.append(f"Xcal:   {Xcal.shape[0]} amostras × {Xcal.shape[1]} variáveis")
    log_parts.append(f"Ycal:   {Ycal.shape[0]} valores")
    log_parts.append(f"Xteste: {Xteste.shape[0]} amostras × {Xteste.shape[1]} variáveis")
    log_parts.append(f"Yteste: {Yteste.shape[0]} valores\n")

    # Otimização
    optimal, opt_log = find_optimal_parameters(Xcal, Ycal, m_values, k_values, progress_cb)
    log_parts.append(opt_log)

    # Avaliação final
    log_parts.append("════════════════════════════════════════════════════════════════")
    log_parts.append("     ETAPA 3: AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
    log_parts.append("════════════════════════════════════════════════════════════════\n")

    # 1. RLM Simples
    log_parts.append("─── 1️⃣  RLM SIMPLES ───")
    t0 = time.perf_counter()
    yPredRLM = perform_rlm(Xcal, Ycal, Xteste)
    ms_rlm = (time.perf_counter() - t0) * 1000
    rmse_rlm, mae_rlm, r2_rlm = calc_metrics(Yteste, yPredRLM)
    log_parts += [f"RMSE = {rmse_rlm:.4f}", f"MAE  = {mae_rlm:.4f}",
                  f"R²   = {r2_rlm:.4f}", f"Tempo: {ms_rlm:.2f} ms\n"]

    # 2. Bagging
    m_bag = optimal["m_bagging"]
    log_parts.append(f"─── 2️⃣  RLM + BAGGING (m* = {m_bag}) ───")
    t1 = time.perf_counter()
    yPredBag = perform_bagging(Xcal, Ycal, Xteste, m_bag, True, 1.0)
    ms_bag = (time.perf_counter() - t1) * 1000
    rmse_bag, mae_bag, r2_bag = calc_metrics(Yteste, yPredBag)
    log_parts += [f"RMSE = {rmse_bag:.4f}", f"MAE  = {mae_bag:.4f}",
                  f"R²   = {r2_bag:.4f}", f"Tempo: {ms_bag:.2f} ms\n"]

    # 3. Subagging
    m_sub = optimal["m_subagging"]
    k_sub = optimal["k_subagging"]
    log_parts.append(f"─── 3️⃣  RLM + SUBAGGING (k* = {k_sub*100:.0f}%, m* = {m_sub}) ───")
    t2 = time.perf_counter()
    yPredSub = perform_bagging(Xcal, Ycal, Xteste, m_sub, False, k_sub)
    ms_sub = (time.perf_counter() - t2) * 1000
    rmse_sub, mae_sub, r2_sub = calc_metrics(Yteste, yPredSub)
    log_parts += [f"RMSE = {rmse_sub:.4f}", f"MAE  = {mae_sub:.4f}",
                  f"R²   = {r2_sub:.4f}", f"Tempo: {ms_sub:.2f} ms\n"]

    # Tabela comparativa
    log_parts.append("════════════════════════════════════════════════════════════════")
    log_parts.append("                 📊 COMPARAÇÃO FINAL 📊")
    log_parts.append("════════════════════════════════════════════════════════════════\n")
    log_parts.append("┌─────────────────────────────────────────────────────────────┐")
    log_parts.append("│                    PARÂMETROS ÓTIMOS                        │")
    log_parts.append("├─────────────────────────────────────────────────────────────┤")
    log_parts.append(f"│ RLM-Bagging:   m* = {m_bag:2d}    (RMSE-CV = {optimal['rmse_cv_bagging']:.4f})           │")
    log_parts.append(f"│ RLM-Subagging: k* = {k_sub*100:.0f}%, m* = {m_sub:2d}  (RMSE-CV = {optimal['rmse_cv_subagging']:.4f})  │")
    log_parts.append("└─────────────────────────────────────────────────────────────┘\n")
    log_parts.append("┌─────────────────────────────────────────────────────────────┐")
    log_parts.append("│              MÉTRICAS NO CONJUNTO DE TESTE                  │")
    log_parts.append("├──────────────────┬──────────┬──────────┬──────────┬─────────┤")
    log_parts.append("│ Técnica          │   RMSE   │   MAE    │    R²    │  Tempo  │")
    log_parts.append("├──────────────────┼──────────┼──────────┼──────────┼─────────┤")
    log_parts.append(f"│ RLM Simples      │ {rmse_rlm:8.4f} │ {mae_rlm:8.4f} │ {r2_rlm:8.4f} │{ms_rlm:6.1f}ms │")
    log_parts.append(f"│ RLM + Bagging    │ {rmse_bag:8.4f} │ {mae_bag:8.4f} │ {r2_bag:8.4f} │{ms_bag:6.1f}ms │")
    log_parts.append(f"│ RLM + Subagging  │ {rmse_sub:8.4f} │ {mae_sub:8.4f} │ {r2_sub:8.4f} │{ms_sub:6.1f}ms │")
    log_parts.append("└──────────────────┴──────────┴──────────┴──────────┴─────────┘\n")

    labels = ["RLM Simples", "RLM + Bagging", "RLM + Subagging"]
    rmses  = [rmse_rlm, rmse_bag, rmse_sub]
    r2s    = [r2_rlm,   r2_bag,   r2_sub]
    times  = [ms_rlm,   ms_bag,   ms_sub]
    log_parts.append(f"✅ Menor RMSE:   {labels[int(np.argmin(rmses))]} ({min(rmses):.4f})")
    log_parts.append(f"✅ Maior R²:     {labels[int(np.argmax(r2s))]} ({max(r2s):.4f})")
    log_parts.append(f"✅ Mais rápido:  {labels[int(np.argmin(times))]} ({min(times):.2f} ms)")
    log_parts.append(f"\n💾 Relatório gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    log_parts.append("════════════════════════════════════════════════════════════════")
    log_parts.append("✓ Análise concluída com protocolo metodologicamente correto!")
    log_parts.append("════════════════════════════════════════════════════════════════")

    return {
        "log": "\n".join(log_parts),
        "optimal": optimal,
        "metrics": {
            "rlm":      {"rmse": rmse_rlm, "mae": mae_rlm, "r2": r2_rlm, "time": ms_rlm},
            "bagging":  {"rmse": rmse_bag, "mae": mae_bag, "r2": r2_bag, "time": ms_bag},
            "subagging":{"rmse": rmse_sub, "mae": mae_sub, "r2": r2_sub, "time": ms_sub},
        },
        "preds": {
            "y_real":      Yteste.tolist(),
            "y_pred_rlm":  yPredRLM.tolist(),
            "y_pred_bag":  yPredBag.tolist(),
            "y_pred_sub":  yPredSub.tolist(),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE UI
# ─────────────────────────────────────────────────────────────────────────────

def parse_int_list(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    result = []
    for p in parts:
        v = int(p)
        if v <= 0:
            raise ValueError(f"Valor inválido de m: '{p}'. Use inteiros positivos.")
        result.append(v)
    return result


def parse_float_list(text):
    parts = [p.strip() for p in text.split(",") if p.strip()]
    result = []
    for p in parts:
        v = float(p)
        if not (0 < v <= 1):
            raise ValueError(f"Valor inválido de k: '{p}'. Use decimais entre 0 e 1.")
        result.append(v)
    return result


def metric_card_html(title, badge, color, metrics):
    return f"""
<div class="metric-card" style="border-left: 5px solid {color}; background:white;
     border-radius:12px; padding:20px; box-shadow:0 2px 12px rgba(0,0,0,0.1);">
  <div style="font-weight:700; color:{color}; font-size:0.9em; margin-bottom:14px;">
    {badge} {title}
  </div>
  <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f0f0f0;">
    <span style="color:#555;font-size:0.9em;">RMSE</span>
    <span style="font-weight:700;color:{color};font-family:monospace;">{metrics['rmse']:.4f}</span>
  </div>
  <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f0f0f0;">
    <span style="color:#555;font-size:0.9em;">MAE</span>
    <span style="font-weight:700;color:{color};font-family:monospace;">{metrics['mae']:.4f}</span>
  </div>
  <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f0f0f0;">
    <span style="color:#555;font-size:0.9em;">R²</span>
    <span style="font-weight:700;color:{color};font-family:monospace;">{metrics['r2']:.4f}</span>
  </div>
  <div style="display:flex;justify-content:space-between;padding:6px 0;">
    <span style="color:#555;font-size:0.9em;">Tempo</span>
    <span style="font-weight:700;color:#666;font-family:monospace;">{metrics['time']:.1f} ms</span>
  </div>
</div>
"""


def build_chart(y_real, y_pred, title, pred_color):
    n = len(y_real)
    x = list(range(n))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_real, mode="lines", name="Referência (Real)",
        line=dict(color="#2196f3", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_pred, mode="lines", name="Predito",
        line=dict(color=pred_color, width=2, dash="dash")
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="black", size=15), x=0, xanchor="left"),
        xaxis_title="Amostra",
        yaxis_title="Valor",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        margin=dict(l=50, r=20, t=60, b=40),
        height=320,
    )
    fig.update_xaxes(
        showgrid=True, gridcolor="#000000",
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#000000",
        title_font=dict(color="black"),
        tickfont=dict(color="black"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

# Cabeçalho
st.markdown("""
<div class="main-header">
  <h1>⚗️ Regressão Espectral</h1>
  <p>RLM Simples · RLM + Bagging · RLM + Subagging com Otimização por Validação Cruzada 5-Fold</p>
</div>
""", unsafe_allow_html=True)

# Abas (equivalente ao Tabs do Vaadin)
tab_config, tab_results, tab_charts = st.tabs(["⚙️  Configuração", "📊  Resultados", "📈  Gráficos"])


# ─────────────────────────────────────────────────────────────────────────────
# ABA CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

with tab_config:
    st.markdown("### 📁 Arquivos de Entrada")

    col1, col2 = st.columns(2)

    # ── strings reutilizáveis ──────────────────────────────────────────────────
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
    # ──────────────────────────────────────────────────────────────────────────

    with col1:
        st.markdown('<div class="upload-hint">🔷 Xcal</div>', unsafe_allow_html=True)
        st.markdown('<div class="upload-desc">Calibração - X (matriz) — .txt · .csv · .xlsx</div>', unsafe_allow_html=True)
        xcal_file = st.file_uploader(
            "Xcal — matriz", type=_TIPOS_MATRIZ, key="xcal",
            label_visibility="collapsed", help=_HELP_MATRIZ
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
            label_visibility="collapsed", help=_HELP_VETOR
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
            label_visibility="collapsed", help=_HELP_MATRIZ
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
            label_visibility="collapsed", help=_HELP_VETOR
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
            help="Inteiros positivos separados por vírgula. Ex: 10, 20, 30, 40, 50"
        )
    with pcol2:
        k_input = st.text_input(
            "Valores de k (subagging)",
            value="0.10, 0.20, 0.30, 0.40, 0.50",
            help="Decimais entre 0 e 1 separados por vírgula. Ex: 0.5, 0.6, 0.7, 0.8"
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

        # Leitura dos arquivos
        try:
            Xcal   = parse_matrix(xcal_file)
            Ycal   = parse_vector(ycal_file)
            Xteste = parse_matrix(xteste_file)
            Yteste = parse_vector(yteste_file)
        except Exception as e:
            st.error(f"❌ Erro ao ler arquivos: {e}")
            st.stop()

        # Progresso
        progress_bar = st.progress(0, text="⏳ Processando análise com otimização por validação cruzada...")

        def update_progress(val):
            progress_bar.progress(min(val * 0.95, 0.95), text="⏳ Otimizando parâmetros...")

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
            st.markdown(metric_card_html("RLM Simples", "─", "#1565c0", metrics["rlm"]),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card_html(
                f"RLM + Bagging (m*={optimal['m_bagging']})", "🅱", "#2e7d32", metrics["bagging"]),
                unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card_html(
                f"RLM + Subagging (k*={optimal['k_subagging']*100:.0f}%, m*={optimal['m_subagging']})",
                "🆂", "#6a1b9a", metrics["subagging"]),
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabela resumida
        st.markdown("### 📋 Tabela Comparativa")
        df_metrics = pd.DataFrame({
            "Técnica":   ["RLM Simples", "RLM + Bagging", "RLM + Subagging"],
            "RMSE":      [metrics["rlm"]["rmse"],      metrics["bagging"]["rmse"],  metrics["subagging"]["rmse"]],
            "MAE":       [metrics["rlm"]["mae"],       metrics["bagging"]["mae"],   metrics["subagging"]["mae"]],
            "R²":        [metrics["rlm"]["r2"],        metrics["bagging"]["r2"],    metrics["subagging"]["r2"]],
            "Tempo (ms)":[metrics["rlm"]["time"],      metrics["bagging"]["time"],  metrics["subagging"]["time"]],
        })
        st.dataframe(
            df_metrics.style.format({"RMSE": "{:.4f}", "MAE": "{:.4f}", "R²": "{:.4f}", "Tempo (ms)": "{:.1f}"}),
            use_container_width=True, hide_index=True
        )

        # Download CSV
        csv = df_metrics.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Baixar métricas como CSV", csv, "metricas.csv", "text/csv")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🖥️ Log de Execução")
        st.code(result["log"], language=None)

        # Download log
        st.download_button("⬇️ Baixar log completo", result["log"].encode("utf-8"),
                           "relatorio.txt", "text/plain")


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

        st.plotly_chart(
            build_chart(preds["y_real"], preds["y_pred_rlm"], "RLM Simples", "#2196f3"),
            use_container_width=True
        )
        st.plotly_chart(
            build_chart(preds["y_real"], preds["y_pred_bag"],
                        f"RLM + Bagging (m*={optimal['m_bagging']})", "#4caf50"),
            use_container_width=True
        )
        st.plotly_chart(
            build_chart(preds["y_real"], preds["y_pred_sub"],
                        f"RLM + Subagging (k*={optimal['k_subagging']*100:.0f}%, m*={optimal['m_subagging']})",
                        "#9c27b0"),
            use_container_width=True
        )

        # Gráfico de dispersão Real × Predito (todos os modelos)
        st.markdown("### 🎯 Dispersão Real × Predito")
        fig_scatter = go.Figure()
        ymin = min(preds["y_real"])
        ymax = max(preds["y_real"])
        fig_scatter.add_trace(go.Scatter(
            x=[ymin, ymax], y=[ymin, ymax], mode="lines",
            name="Ideal (y=x)", line=dict(color="gray", dash="dot")
        ))
        for label, key, color in [
            ("RLM Simples",     "y_pred_rlm", "#2196f3"),
            ("RLM + Bagging",   "y_pred_bag", "#4caf50"),
            ("RLM + Subagging", "y_pred_sub", "#9c27b0"),
        ]:
            fig_scatter.add_trace(go.Scatter(
                x=preds["y_real"], y=preds[key], mode="markers",
                name=label, marker=dict(color=color, size=6, opacity=0.7)
            ))
        fig_scatter.update_layout(
            title=dict(text="Dispersão Real × Predito", font=dict(color="black", size=15), x=0, xanchor="left"),
            xaxis_title="Valor Real", yaxis_title="Valor Predito",
            plot_bgcolor="#fafafa", paper_bgcolor="white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=50, r=20, t=60, b=40),
            height=420
        )
        fig_scatter.update_xaxes(
            showgrid=True, gridcolor="#000000",
            title_font=dict(color="black"),
            tickfont=dict(color="black"),
        )
        fig_scatter.update_yaxes(
            showgrid=True, gridcolor="#000000",
            title_font=dict(color="black"),
            tickfont=dict(color="black"),
        )
        st.plotly_chart(fig_scatter, use_container_width=True)