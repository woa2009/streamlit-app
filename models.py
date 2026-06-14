"""
models.py — Motor Matemático
Regressão Espectral: RLM, Bagging e Subagging

Contém exclusivamente funções puras (NumPy/Pandas).
Nenhuma dependência de Streamlit ou de bibliotecas visuais.

Autor: Wagner Oliveira de Araujo
Versão: 06
"""

import io
import numpy as np
import pandas as pd
import time
from datetime import datetime


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
) -> tuple[dict, str]:
    """
    Grid search com CV 5-Fold para encontrar m* (Bagging) e k*, m* (Subagging).

    Retorna
    -------
    (optimal: dict, log: str)
    """
    log = []
    log.append("════════════════════════════════════════════════════════════════")
    log.append("     ETAPA 2: OTIMIZAÇÃO DE PARÂMETROS (Validação Cruzada 5-Fold | solver: lstsq)")
    log.append("════════════════════════════════════════════════════════════════\n")
    log.append(f"📋 Valores de m testados: {m_values}")
    log.append(f"📋 Valores de k testados: {k_values}\n")

    total_steps = len(m_values) + len(m_values) * len(k_values)
    step = 0

    # --- Otimiza Bagging ---
    log.append("🔍 OTIMIZANDO RLM-BAGGING (testando m):")
    log.append("─────────────────────────────────────────")
    best_bag_rmse, best_bag_m, best_bag_rmse_cv = float("inf"), m_values[0], 0.0

    for m in m_values:
        mean_r, std_r = run_cross_validation(Xcal, Ycal, m, True, 1.0)
        log.append(f"   m={m:3d} → RMSE-CV = {mean_r:.4f} ± {std_r:.4f}")
        if mean_r < best_bag_rmse:
            best_bag_rmse, best_bag_m, best_bag_rmse_cv = mean_r, m, mean_r
        step += 1
        if progress_callback:
            progress_callback(step / total_steps)

    log.append(f"\n✅ Melhor Bagging: m* = {best_bag_m} (RMSE-CV = {best_bag_rmse:.4f})\n")

    # --- Otimiza Subagging ---
    log.append("🔍 OTIMIZANDO RLM-SUBAGGING (testando k × m):")
    log.append("─────────────────────────────────────────────────")
    best_sub_rmse, best_sub_m, best_sub_k, best_sub_rmse_cv = float("inf"), m_values[0], k_values[0], 0.0

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
) -> dict:
    """
    Executa a análise completa: otimização por CV + avaliação final dos 3 modelos.

    Retorna
    -------
    dict com chaves: log, optimal, metrics, preds
    """
    log_parts = []
    log_parts.append("════════════════════════════════════════════════════════════════")
    log_parts.append("   ANÁLISE COM OTIMIZAÇÃO POR VALIDAÇÃO CRUZADA 5-FOLD  |  solver: lstsq")
    log_parts.append("════════════════════════════════════════════════════════════════\n")
    log_parts.append(f"📋 Valores de m (bags): {m_values}")
    log_parts.append(f"📋 Valores de k (subagging): {k_values}\n")
    log_parts.append("=== DIMENSÕES DETECTADAS ===")
    log_parts.append(f"Xcal:   {Xcal.shape[0]} amostras × {Xcal.shape[1]} variáveis")
    log_parts.append(f"Ycal:   {Ycal.shape[0]} valores")
    log_parts.append(f"Xteste: {Xteste.shape[0]} amostras × {Xteste.shape[1]} variáveis")
    log_parts.append(f"Yteste: {Yteste.shape[0]} valores\n")

    # Etapa 2: Otimização
    optimal, opt_log = find_optimal_parameters(Xcal, Ycal, m_values, k_values, progress_cb)
    log_parts.append(opt_log)

    # Etapa 3: Avaliação final no conjunto de teste
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

    # 2. Bagging com m* ótimo
    m_bag = optimal["m_bagging"]
    log_parts.append(f"─── 2️⃣  RLM + BAGGING (m* = {m_bag}) ───")
    t1 = time.perf_counter()
    yPredBag = perform_bagging(Xcal, Ycal, Xteste, m_bag, True, 1.0)
    ms_bag = (time.perf_counter() - t1) * 1000
    rmse_bag, mae_bag, r2_bag = calc_metrics(Yteste, yPredBag)
    log_parts += [f"RMSE = {rmse_bag:.4f}", f"MAE  = {mae_bag:.4f}",
                  f"R²   = {r2_bag:.4f}", f"Tempo: {ms_bag:.2f} ms\n"]

    # 3. Subagging com k* e m* ótimos
    m_sub = optimal["m_subagging"]
    k_sub = optimal["k_subagging"]
    log_parts.append(f"─── 3️⃣  RLM + SUBAGGING (k* = {k_sub*100:.0f}%, m* = {m_sub}) ───")
    t2 = time.perf_counter()
    yPredSub = perform_bagging(Xcal, Ycal, Xteste, m_sub, False, k_sub)
    ms_sub = (time.perf_counter() - t2) * 1000
    rmse_sub, mae_sub, r2_sub = calc_metrics(Yteste, yPredSub)
    log_parts += [f"RMSE = {rmse_sub:.4f}", f"MAE  = {mae_sub:.4f}",
                  f"R²   = {r2_sub:.4f}", f"Tempo: {ms_sub:.2f} ms\n"]

    # Tabela comparativa no log
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
        "log":     "\n".join(log_parts),
        "optimal": optimal,
        "metrics": {
            "rlm":      {"rmse": rmse_rlm, "mae": mae_rlm, "r2": r2_rlm, "time": ms_rlm},
            "bagging":  {"rmse": rmse_bag, "mae": mae_bag, "r2": r2_bag, "time": ms_bag},
            "subagging":{"rmse": rmse_sub, "mae": mae_sub, "r2": r2_sub, "time": ms_sub},
        },
        "preds": {
            "y_real":     Yteste.tolist(),
            "y_pred_rlm": yPredRLM.tolist(),
            "y_pred_bag": yPredBag.tolist(),
            "y_pred_sub": yPredSub.tolist(),
        },
    }
