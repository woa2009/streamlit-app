"""
components.py — Fábrica de Componentes Visuais
Regressão Espectral: RLM, Bagging e Subagging

Contém exclusivamente funções de construção de gráficos (Plotly)
e componentes HTML de interface. Nenhuma lógica matemática aqui.

Autor: Wagner Oliveira de Araujo
"""

import plotly.graph_objects as go


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES VISUAIS
# ─────────────────────────────────────────────────────────────────────────────

COR_RLM       = "#2196f3"   # Azul
COR_BAGGING   = "#4caf50"   # Verde
COR_SUBAGGING = "#9c27b0"   # Roxo

_GRID_COLOR = "#000000"
_PLOT_BG    = "#fafafa"
_PAPER_BG   = "white"

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor=_GRID_COLOR,
    title_font=dict(color="black"),
    tickfont=dict(color="black"),
)

_LAYOUT_BASE = dict(
    plot_bgcolor=_PLOT_BG,
    paper_bgcolor=_PAPER_BG,
    margin=dict(l=50, r=20, t=60, b=40),
)

# Estilo de legenda vertical (canto superior esquerdo, dentro do gráfico)
_LEGEND_STYLE = dict(
    orientation="v",
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01,
    bgcolor="white",
    bordercolor="black",
    borderwidth=1,
    font=dict(color="black"),
)


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO DE LINHA: REAL vs. PREDITO (por modelo)
# ─────────────────────────────────────────────────────────────────────────────

def build_chart(
    y_real: list,
    y_pred: list,
    title: str,
    pred_color: str,
) -> go.Figure:
    """
    Gráfico de linha sobreposto: valores reais (azul) × preditos (colorido por modelo).

    Parâmetros
    ----------
    y_real      : lista de valores reais (Yteste)
    y_pred      : lista de valores preditos pelo modelo
    title       : título exibido no gráfico
    pred_color  : cor da linha predita (hex)
    """
    n = len(y_real)
    x = list(range(n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y_real,
        mode="lines",
        name="Valor Real",
        line=dict(color=COR_RLM, width=2),
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y_pred,
        mode="lines",
        name="Valor Predito",
        line=dict(color=pred_color, width=2, dash="dash"),
        showlegend=True,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="black", size=15), x=0, xanchor="left"),
        xaxis_title="Amostra",
        yaxis_title="Concentração",
        legend=_LEGEND_STYLE,
        height=320,
        **_LAYOUT_BASE,
    )
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO DE DISPERSÃO: REAL × PREDITO (todos os modelos juntos)
# ─────────────────────────────────────────────────────────────────────────────

def build_scatter_chart(preds: dict) -> go.Figure:
    """
    Dispersão Real × Predito com os 3 modelos sobrepostos e linha ideal (y=x).

    Parâmetros
    ----------
    preds : dict com chaves y_real, y_pred_rlm, y_pred_bag, y_pred_sub
    """
    y_real = preds["y_real"]
    ymin   = min(y_real)
    ymax   = max(y_real)

    fig = go.Figure()

    # Linha ideal y = x
    fig.add_trace(go.Scatter(
        x=[ymin, ymax], y=[ymin, ymax],
        mode="lines",
        name="Ideal (y=x)",
        line=dict(color="gray", dash="dot"),
        showlegend=True,
    ))

    # Pontos por modelo
    for label, key, color in [
        ("RLM Simples",     "y_pred_rlm", COR_RLM),
        ("RLM + Bagging",   "y_pred_bag", COR_BAGGING),
        ("RLM + Subagging", "y_pred_sub", COR_SUBAGGING),
    ]:
        fig.add_trace(go.Scatter(
            x=y_real,
            y=preds[key],
            mode="markers",
            name=label,
            marker=dict(color=color, size=6, opacity=0.7),
            showlegend=True,
        ))

    fig.update_layout(
        title=dict(
            text="Dispersão Real × Predito",
            font=dict(color="black", size=15),
            x=0, xanchor="left",
        ),
        xaxis_title="Valor Real",
        yaxis_title="Valor Predito",
        legend=_LEGEND_STYLE,
        height=420,
        **_LAYOUT_BASE,
    )
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CARD HTML DE MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def metric_card_html(title: str, badge: str, color: str, metrics: dict) -> str:
    """
    Gera HTML de um card de métricas (RMSE, MAE, R², Tempo).

    Parâmetros
    ----------
    title   : nome do modelo
    badge   : emoji/ícone exibido ao lado do título
    color   : cor hex da borda e dos valores
    metrics : dict com chaves rmse, mae, r2, time
    """
    def row(label, fmt, val_color=None):
        vc = val_color or color
        return (
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:6px 0;border-bottom:1px solid #f0f0f0;">'
            f'<span style="color:#555;font-size:0.9em;">{label}</span>'
            f'<span style="font-weight:700;color:{vc};font-family:monospace;">{fmt}</span>'
            f'</div>'
        )

    return f"""
<div style="background:white;border-left:5px solid {color};border-radius:12px;
     padding:20px;box-shadow:0 2px 12px rgba(0,0,0,0.1);height:100%;">
  <div style="font-weight:700;color:{color};font-size:0.9em;margin-bottom:14px;">
    {badge} {title}
  </div>
  {row("RMSE",  f"{metrics['rmse']:.4f}")}
  {row("MAE",   f"{metrics['mae']:.4f}")}
  {row("R²",    f"{metrics['r2']:.4f}")}
  {row("Tempo", f"{metrics['time']:.1f} ms", "#666")}
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# CSS GLOBAL DA APLICAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

APP_CSS = """
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 100%);
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8em; }
    .main-header p  { color: rgba(255,255,255,0.85); margin: 4px 0 0 0; font-size: 0.9em; }

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
"""

HEADER_HTML = """
<div class="main-header">
  <h1>⚗️ Regressão Espectral</h1>
  <p>RLM Simples · RLM + Bagging · RLM + Subagging com Otimização por Validação Cruzada 5-Fold</p>
</div>
"""
