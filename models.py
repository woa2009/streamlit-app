"""
models.py — Motor Matemático
Regressão Espectral: RLM, Bagging e Subagging

Contém exclusivamente funções puras (NumPy/Pandas).
Nenhuma dependência de Streamlit ou de bibliotecas visuais.

Autor: Wagner Oliveira de Araujo
Versão: 8a
"""

import io
import numpy as np
import pandas as pd
import time
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# TRADUÇÕES DO LOG
# ─────────────────────────────────────────────────────────────────────────────

_TRADUCOES_LOG = {
    "pt": {
        # find_optimal_parameters
        "etapa2_titulo":        "     ETAPA 2: OTIMIZAÇÃO DE PARÂMETROS (Validação Cruzada 5-Fold | solver: lstsq)",
        "m_testados":           "📋 Valores de m testados: ",
        "k_testados":           "📋 Valores de k testados: ",
        "otim_bagging":         "🔍 OTIMIZANDO RLM-BAGGING (testando m):",
        "otim_subagging":       "🔍 OTIMIZANDO RLM-SUBAGGING (testando k × m):",
        "testando_k":           "   Testando k=",
        "melhor_bagging":       "✅ Melhor Bagging: m* = ",
        "melhor_subagging":     "✅ Melhor Subagging: k* = ",
        "rmse_cv":              "RMSE-CV = ",
        # run_full_analysis — cabeçalho
        "cabecalho":            "   ANÁLISE COM OTIMIZAÇÃO POR VALIDAÇÃO CRUZADA 5-FOLD  |  solver: lstsq",
        "m_bags":               "📋 Valores de m (bags): ",
        "k_sub":                "📋 Valores de k (subagging): ",
        "dimensoes":            "=== DIMENSÕES DETECTADAS ===",
        "amostras":             " amostras × ",
        "variaveis":            " variáveis",
        "valores":              " valores",
        # run_full_analysis — etapa 3
        "etapa3_titulo":        "     ETAPA 3: AVALIAÇÃO FINAL NO CONJUNTO DE TESTE",
        "rlm_simples":          "─── 1️⃣  RLM SIMPLES ───",
        "tempo_label":          "Tempo: ",
        # run_full_analysis — tabela comparativa
        "comp_final":           "                 📊 COMPARAÇÃO FINAL 📊",
        "params_otimos":        "│                    PARÂMETROS ÓTIMOS                        │",
        "metricas_teste":       "│              MÉTRICAS NO CONJUNTO DE TESTE                  │",
        "col_tecnica":          "│ Técnica          │   RMSE   │   MAE    │    R²    │  Tempo  │",
        "nome_rlm":             "RLM Simples",
        "nome_bagging":         "RLM + Bagging",
        "nome_subagging":       "RLM + Subagging",
        "row_rlm":              "│ RLM Simples      │",
        "row_bagging":          "│ RLM + Bagging    │",
        "row_subagging":        "│ RLM + Subagging  │",
        "menor_rmse":           "✅ Menor RMSE:   ",
        "maior_r2":             "✅ Maior R²:     ",
        "mais_rapido":          "✅ Mais rápido:  ",
        "relatorio_gerado":     "💾 Relatório gerado em: ",
        "analise_concluida":    "✓ Análise concluída com protocolo metodologicamente correto!",
    },
    "en": {
        # find_optimal_parameters
        "etapa2_titulo":        "     STEP 2: PARAMETER OPTIMIZATION (5-Fold Cross-Validation | solver: lstsq)",
        "m_testados":           "📋 Tested m values: ",
        "k_testados":           "📋 Tested k values: ",
        "otim_bagging":         "🔍 OPTIMIZING MLR-BAGGING (testing m):",
        "otim_subagging":       "🔍 OPTIMIZING MLR-SUBAGGING (testing k × m):",
        "testando_k":           "   Testing k=",
        "melhor_bagging":       "✅ Best Bagging: m* = ",
        "melhor_subagging":     "✅ Best Subagging: k* = ",
        "rmse_cv":              "RMSE-CV = ",
        # run_full_analysis — cabeçalho
        "cabecalho":            "   ANALYSIS WITH 5-FOLD CROSS-VALIDATION OPTIMIZATION  |  solver: lstsq",
        "m_bags":               "📋 m values (bags): ",
        "k_sub":                "📋 k values (subagging): ",
        "dimensoes":            "=== DETECTED DIMENSIONS ===",
        "amostras":             " samples × ",
        "variaveis":            " variables",
        "valores":              " values",
        # run_full_analysis — etapa 3
        "etapa3_titulo":        "     STEP 3: FINAL EVALUATION ON TEST SET",
        "rlm_simples":          "─── 1️⃣  SIMPLE MLR ───",
        "tempo_label":          "Time: ",
        # run_full_analysis — tabela comparativa
        "comp_final":           "                 📊 FINAL COMPARISON 📊",
        "params_otimos":        "│                    OPTIMAL PARAMETERS                        │",
        "metricas_teste":       "│              METRICS ON TEST SET                             │",
        "col_tecnica":          "│ Method           │   RMSE   │   MAE    │    R²    │  Time   │",
        "nome_rlm":             "Simple MLR",
        "nome_bagging":         "MLR + Bagging",
        "nome_subagging":       "MLR + Subagging",
        "row_rlm":              "│ Simple MLR       │",
        "row_bagging":          "│ MLR + Bagging    │",
        "row_subagging":        "│ MLR + Subagging  │",
        "menor_rmse":           "✅ Lowest RMSE:  ",
        "maior_r2":             "✅ Highest R²:   ",
        "mais_rapido":          "✅ Fastest:      ",
        "relatorio_gerado":     "💾 Report generated at: ",
        "analise_concluida":    "✓ Analysis completed with methodologically correct protocol!",
    },
}


def _tl(lang: str, chave: str) -> str:
    """Retorna a tradução de uma chave do log para o idioma especificado."""
    return _TRADUCOES_LOG.get(lang, _TRADUCOES_LOG["pt"])[chave]


# ─────────────────────────────────────────────────────────────────────────────
# PARSE DE ARQUIVOS  (suporta .txt · .csv · .xlsx)
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

    # .txt (comportamento padrão, suporta decimal BR)
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


# ─────────────────────────────────────────────────────────────────────────────
# MODELOS DE REGRESSÃO
# ─────────────────────────────────────────────────────────────────────────────

def perform_rlm(Xcal: np.ndarray, Ycal: np.ndarray, Xteste: np.ndarray) -> np.ndarray:
    """RLM simples via mínimos quadrados (lstsq).

    Equivalente à pseudoinversa de Moore-Penrose, porém 3-5x mais rápido para
    matrizes de alta dimensionalidade (ex.: espectros NIR/IR com milhares de variáveis),
    pois usa decomposição QR/SVD parcial em vez do SVD completo do np.linalg.pinv.
    """
    beta, _, _, _ = np.linalg.lstsq(Xcal, Ycal, rcond=None)
    return Xteste @ beta


def perform_bagging(
    Xcal: np.ndarray,
    Ycal: np.ndarray,
    Xteste: np.ndarray,
    num_bags: int,
    with_replacement: bool,
    fraction: float,
    seed: int = 42,
) -> np.ndarray:
    """
    RLM com Bagging (with_replacement=True) ou Subagging (with_replacement=False).

    O solver np.linalg.lstsq é usado internamente em cada bag — equivalente à
    pseudoinversa de Moore-Penrose, mas significativamente mais rápido para
    matrizes espectrais de alta dimensionalidade (linhas < colunas).

    Parâmetros
    ----------
    num_bags        : número de ensembles (m)
    with_replacement: True → Bagging (bootstrap); False → Subagging
    fraction        : fração de amostras por bag (k), usado apenas no Subagging
    seed            : semente para reprodutibilidade
    """
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
        beta, _, _, _ = np.linalg.lstsq(Xbag, Ybag, rcond=None)  # lstsq: 3-5x mais rápido que pinv
        Ypred += (Xteste @ beta) / num_bags

    return Ypred


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def calc_metrics(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float, float]:
    """
    Calcula RMSE, MAE e R².

    Retorna
    -------
    (rmse, mae, r2)
    """
    err    = predicted - actual
    rmse   = float(np.sqrt(np.mean(err ** 2)))
    mae    = float(np.mean(np.abs(err)))
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    ss_res = np.sum(err ** 2)
    r2     = float(1 - (ss_res / ss_tot)) if ss_tot != 0 else 0.0
    return rmse, mae, r2


# ─────────────────────────────────────────────────────────────────────────────
# VALIDAÇÃO CRUZADA
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(
    Xcal: np.ndarray,
    Ycal: np.ndarray,
    m: int,
    with_replacement: bool,
    fraction: float,
    n_folds: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Validação cruzada k-fold sobre perform_bagging.

    Retorna
    -------
    (mean_rmse, std_rmse)
    """
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
        rmse_list.append(float(np.sqrt(np.mean((Ypred - Yval) ** 2))))

    return float(np.mean(rmse_list)), float(np.std(rmse_list))


# ─────────────────────────────────────────────────────────────────────────────
# OTIMIZAÇÃO DE HIPERPARÂMETROS
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_parameters(
    Xcal: np.ndarray,
    Ycal: np.ndarray,
    m_values: list[int],
    k_values: list[float],
    progress_callback=None,
    lang: str = "pt",
) -> tuple[dict, str]:
    """
    Grid search com CV 5-Fold para encontrar m* (Bagging) e k*, m* (Subagging).

    Parâmetros
    ----------
    lang : idioma do log gerado — 'pt' (padrão) ou 'en'

    Retorna
    -------
    (optimal: dict, log: str)
    """
    L = lambda chave: _tl(lang, chave)  # noqa: E731

    log = []
    log.append("════════════════════════════════════════════════════════════════")
    log.append(L("etapa2_titulo"))
    log.append("════════════════════════════════════════════════════════════════\n")
    log.append(L("m_testados") + str(m_values))
    log.append(L("k_testados") + str(k_values) + "\n")

    total_steps = len(m_values) + len(m_values) * len(k_values)
    step = 0

    # --- Otimiza Bagging ---
    log.append(L("otim_bagging"))
    log.append("─────────────────────────────────────────")
    best_bag_rmse, best_bag_m, best_bag_rmse_cv = float("inf"), m_values[0], 0.0

    for m in m_values:
        mean_r, std_r = run_cross_validation(Xcal, Ycal, m, True, 1.0)
        log.append(f"   m={m:3d} → {L('rmse_cv')}{mean_r:.4f} ± {std_r:.4f}")
        if mean_r < best_bag_rmse:
            best_bag_rmse, best_bag_m, best_bag_rmse_cv = mean_r, m, mean_r
        step += 1
        if progress_callback:
            progress_callback(step / total_steps)

    log.append(f"\n{L('melhor_bagging')}{best_bag_m} ({L('rmse_cv')}{best_bag_rmse:.4f})\n")

    # --- Otimiza Subagging ---
    log.append(L("otim_subagging"))
    log.append("─────────────────────────────────────────────────")
    best_sub_rmse, best_sub_m, best_sub_k, best_sub_rmse_cv = float("inf"), m_values[0], k_values[0], 0.0

    for k in k_values:
        log.append(f"\n{L('testando_k')}{k*100:.0f}%:")
        for m in m_values:
            mean_r, std_r = run_cross_validation(Xcal, Ycal, m, False, k)
            log.append(f"      m={m:3d} → {L('rmse_cv')}{mean_r:.4f} ± {std_r:.4f}")
            if mean_r < best_sub_rmse:
                best_sub_rmse, best_sub_m, best_sub_k, best_sub_rmse_cv = mean_r, m, k, mean_r
            step += 1
            if progress_callback:
                progress_callback(step / total_steps)

    log.append(
        f"\n{L('melhor_subagging')}{best_sub_k*100:.0f}%, m* = {best_sub_m} "
        f"({L('rmse_cv')}{best_sub_rmse:.4f})\n"
    )

    optimal = {
        "m_bagging":         best_bag_m,
        "m_subagging":       best_sub_m,
        "k_subagging":       best_sub_k,
        "rmse_cv_bagging":   best_bag_rmse_cv,
        "rmse_cv_subagging": best_sub_rmse_cv,
    }
    return optimal, "\n".join(log)


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISE COMPLETA
# ─────────────────────────────────────────────────────────────────────────────

def run_full_analysis(
    Xcal: np.ndarray,
    Ycal: np.ndarray,
    Xteste: np.ndarray,
    Yteste: np.ndarray,
    m_values: list[int],
    k_values: list[float],
    progress_cb=None,
    lang: str = "pt",
) -> dict:
    """
    Executa a análise completa: otimização por CV + avaliação final dos 3 modelos.

    Parâmetros
    ----------
    lang : idioma do log gerado — 'pt' (padrão) ou 'en'

    Retorna
    -------
    dict com chaves: log, optimal, metrics, preds
    """
    L = lambda chave: _tl(lang, chave)  # noqa: E731

    log_parts = []
    log_parts.append("════════════════════════════════════════════════════════════════")
    log_parts.append(L("cabecalho"))
    log_parts.append("════════════════════════════════════════════════════════════════\n")
    log_parts.append(L("m_bags") + str(m_values))
    log_parts.append(L("k_sub") + str(k_values) + "\n")
    log_parts.append(L("dimensoes"))
    log_parts.append(f"Xcal:   {Xcal.shape[0]}{L('amostras')}{Xcal.shape[1]}{L('variaveis')}")
    log_parts.append(f"Ycal:   {Ycal.shape[0]}{L('valores')}")
    log_parts.append(f"Xteste: {Xteste.shape[0]}{L('amostras')}{Xteste.shape[1]}{L('variaveis')}")
    log_parts.append(f"Yteste: {Yteste.shape[0]}{L('valores')}\n")

    # Etapa 2: Otimização
    optimal, opt_log = find_optimal_parameters(
        Xcal, Ycal, m_values, k_values, progress_cb, lang
    )
    log_parts.append(opt_log)

    # Etapa 3: Avaliação final no conjunto de teste
    log_parts.append("════════════════════════════════════════════════════════════════")
    log_parts.append(L("etapa3_titulo"))
    log_parts.append("════════════════════════════════════════════════════════════════\n")

    # 1. RLM Simples
    log_parts.append(L("rlm_simples"))
    t0 = time.perf_counter()
    yPredRLM = perform_rlm(Xcal, Ycal, Xteste)
    ms_rlm = (time.perf_counter() - t0) * 1000
    rmse_rlm, mae_rlm, r2_rlm = calc_metrics(Yteste, yPredRLM)
    log_parts += [
        f"RMSE = {rmse_rlm:.4f}",
        f"MAE  = {mae_rlm:.4f}",
        f"R²   = {r2_rlm:.4f}",
        f"{L('tempo_label')}{ms_rlm:.2f} ms\n",
    ]

    # 2. Bagging com m* ótimo
    m_bag = optimal["m_bagging"]
    log_parts.append(f"─── 2️⃣  {L('nome_bagging')} (m* = {m_bag}) ───")
    t1 = time.perf_counter()
    yPredBag = perform_bagging(Xcal, Ycal, Xteste, m_bag, True, 1.0)
    ms_bag = (time.perf_counter() - t1) * 1000
    rmse_bag, mae_bag, r2_bag = calc_metrics(Yteste, yPredBag)
    log_parts += [
        f"RMSE = {rmse_bag:.4f}",
        f"MAE  = {mae_bag:.4f}",
        f"R²   = {r2_bag:.4f}",
        f"{L('tempo_label')}{ms_bag:.2f} ms\n",
    ]

    # 3. Subagging com k* e m* ótimos
    m_sub = optimal["m_subagging"]
    k_sub = optimal["k_subagging"]
    log_parts.append(f"─── 3️⃣  {L('nome_subagging')} (k* = {k_sub*100:.0f}%, m* = {m_sub}) ───")
    t2 = time.perf_counter()
    yPredSub = perform_bagging(Xcal, Ycal, Xteste, m_sub, False, k_sub)
    ms_sub = (time.perf_counter() - t2) * 1000
    rmse_sub, mae_sub, r2_sub = calc_metrics(Yteste, yPredSub)
    log_parts += [
        f"RMSE = {rmse_sub:.4f}",
        f"MAE  = {mae_sub:.4f}",
        f"R²   = {r2_sub:.4f}",
        f"{L('tempo_label')}{ms_sub:.2f} ms\n",
    ]

    # Tabela comparativa no log
    log_parts.append("════════════════════════════════════════════════════════════════")
    log_parts.append(L("comp_final"))
    log_parts.append("════════════════════════════════════════════════════════════════\n")
    log_parts.append("┌─────────────────────────────────────────────────────────────┐")
    log_parts.append(L("params_otimos"))
    log_parts.append("├─────────────────────────────────────────────────────────────┤")
    log_parts.append(f"│ {L('nome_bagging')+':':<15} m* = {m_bag:2d}    ({L('rmse_cv')}{optimal['rmse_cv_bagging']:.4f})           │")
    log_parts.append(f"│ {L('nome_subagging')+':':<15} k* = {k_sub*100:.0f}%, m* = {m_sub:2d}  ({L('rmse_cv')}{optimal['rmse_cv_subagging']:.4f})  │")
    log_parts.append("└─────────────────────────────────────────────────────────────┘\n")
    log_parts.append("┌─────────────────────────────────────────────────────────────┐")
    log_parts.append(L("metricas_teste"))
    log_parts.append("├──────────────────┬──────────┬──────────┬──────────┬─────────┤")
    log_parts.append(L("col_tecnica"))
    log_parts.append("├──────────────────┼──────────┼──────────┼──────────┼─────────┤")
    log_parts.append(f"{L('row_rlm')} {rmse_rlm:8.4f} │ {mae_rlm:8.4f} │ {r2_rlm:8.4f} │{ms_rlm:6.1f}ms │")
    log_parts.append(f"{L('row_bagging')} {rmse_bag:8.4f} │ {mae_bag:8.4f} │ {r2_bag:8.4f} │{ms_bag:6.1f}ms │")
    log_parts.append(f"{L('row_subagging')} {rmse_sub:8.4f} │ {mae_sub:8.4f} │ {r2_sub:8.4f} │{ms_sub:6.1f}ms │")
    log_parts.append("└──────────────────┴──────────┴──────────┴──────────┴─────────┘\n")

    nomes  = [L("nome_rlm"), L("nome_bagging"), L("nome_subagging")]
    rmses  = [rmse_rlm, rmse_bag, rmse_sub]
    r2s    = [r2_rlm,   r2_bag,   r2_sub]
    times  = [ms_rlm,   ms_bag,   ms_sub]
    log_parts.append(f"{L('menor_rmse')}{nomes[int(np.argmin(rmses))]} ({min(rmses):.4f})")
    log_parts.append(f"{L('maior_r2')}{nomes[int(np.argmax(r2s))]} ({max(r2s):.4f})")
    log_parts.append(f"{L('mais_rapido')}{nomes[int(np.argmin(times))]} ({min(times):.2f} ms)")
    log_parts.append(f"\n💾 {L('relatorio_gerado')}{datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    log_parts.append("════════════════════════════════════════════════════════════════")
    log_parts.append(L("analise_concluida"))
    log_parts.append("════════════════════════════════════════════════════════════════")

    return {
        "log":     "\n".join(log_parts),
        "optimal": optimal,
        "metrics": {
            "rlm":       {"rmse": rmse_rlm, "mae": mae_rlm, "r2": r2_rlm, "time": ms_rlm},
            "bagging":   {"rmse": rmse_bag, "mae": mae_bag, "r2": r2_bag, "time": ms_bag},
            "subagging": {"rmse": rmse_sub, "mae": mae_sub, "r2": r2_sub, "time": ms_sub},
        },
        "preds": {
            "y_real":     Yteste.tolist(),
            "y_pred_rlm": yPredRLM.tolist(),
            "y_pred_bag": yPredBag.tolist(),
            "y_pred_sub": yPredSub.tolist(),
        },
    }
