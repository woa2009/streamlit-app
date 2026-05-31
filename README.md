# ⚗️ Regressão Espectral — Streamlit

Conversão do projeto **Vaadin Flow (Java)** para **Streamlit (Python)**.

Implementa RLM Simples, RLM + Bagging e RLM + Subagging com otimização
de hiperparâmetros por validação cruzada 5-fold.

Autor: Wagner Oliveira de Araujo

## 🗂️ Estrutura do Projeto

```
.
├── app.py            # Orquestrador: layout Streamlit, abas e interação com o usuário
├── models.py         # Motor matemático: parse de arquivos, regressão, métricas e CV
├── components.py     # Fábrica visual: gráficos Plotly, cards HTML e CSS global
└── requirements.txt
```

| Arquivo          | Responsabilidade                                                                 | Dependências externas        |
|------------------|----------------------------------------------------------------------------------|------------------------------|
| `app.py`         | `st.tabs`, `st.file_uploader`, `st.button`, barra de progresso, `session_state` | `streamlit`, `pandas`        |
| `models.py`      | `parse_matrix`, `parse_vector`, `perform_rlm`, `perform_bagging`, `run_full_analysis` | `numpy`, `pandas`, `openpyxl` |
| `components.py`  | `build_chart`, `build_scatter_chart`, `metric_card_html`, `APP_CSS`, `HEADER_HTML` | `plotly`                     |

> **Princípio de separação de responsabilidades:** `models.py` não importa `streamlit` nem `plotly`;
> `components.py` não importa `streamlit` nem `numpy`; `app.py` não contém nenhuma função matemática.

## 🚀 Como rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Acesse em: `http://localhost:8501`

> `requirements.txt` inclui `openpyxl>=3.1.0`, necessário para leitura de arquivos `.xlsx`.

## ☁️ Deploy no Streamlit Community Cloud (gratuito)

1. Suba este repositório no GitHub
2. Acesse https://share.streamlit.io
3. Clique em **New app** → selecione o repositório e o arquivo `app.py`
4. Clique em **Deploy**

## 📁 Formato dos arquivos de entrada

Os quatro arquivos aceitam **três formatos**: `.txt`, `.csv` e `.xlsx`.

| Arquivo                | Tipo   | Regras |
|------------------------|--------|--------|
| Xcal `.txt/csv/xlsx`   | Matriz | Colunas por espaço (`.txt`), vírgula ou `;` (`.csv`), planilha sem cabeçalho (`.xlsx`) |
| Ycal `.txt/csv/xlsx`   | Vetor  | Um valor por linha; primeira coluna em `.csv`/`.xlsx` |
| Xteste `.txt/csv/xlsx` | Matriz | Igual ao Xcal |
| Yteste `.txt/csv/xlsx` | Vetor  | Igual ao Ycal |

### Regras por formato

**`.txt`**
- Matrizes: colunas separadas por **espaços simples**, uma amostra por linha.
- Vetores: **um valor por linha**, sem separadores adicionais.
- Decimal: ponto (`.`) ou vírgula (`,` — padrão BR).

**`.csv`**
- Separador de coluna detectado automaticamente: **vírgula** (`,`) ou **ponto-e-vírgula** (`;`).
- Se o separador for `;`, vírgulas são tratadas como separador decimal (padrão BR).
- Não misture separador de coluna e decimal no mesmo arquivo.

**`.xlsx`**
- Primeira planilha utilizada, sem linha de cabeçalho.
- Matrizes: cada linha = uma amostra; colunas = variáveis.
- Vetores: primeira coluna utilizada.

A interface exibe, em cada campo de upload, um tooltip `?` e um expander
**"ℹ️ Como formatar o arquivo"** com exemplos para cada formato.

## 🔄 Correspondência Vaadin → Streamlit

| Vaadin (Java)                  | Streamlit / Python                          | Arquivo           |
|--------------------------------|---------------------------------------------|-------------------|
| `@Route("")`                   | `st.set_page_config()`                      | `app.py`          |
| `Tabs`                         | `st.tabs()`                                 | `app.py`          |
| `Upload + MemoryBuffer`        | `st.file_uploader()`                        | `app.py`          |
| `TextField`                    | `st.text_input()`                           | `app.py`          |
| `Button + ClickListener`       | `st.button()`                               | `app.py`          |
| `ProgressBar`                  | `st.progress()` + callback                  | `app.py`          |
| `Grid<Entidade>`               | `st.dataframe()`                            | `app.py`          |
| `Notification`                 | `st.success()` / `st.error()`               | `app.py`          |
| `HttpSession` (estado)         | `st.session_state["result"]`                | `app.py`          |
| SVG inline (gráficos)          | `plotly` interativo                         | `components.py`   |
| Estilos CSS inline (Vaadin)    | `APP_CSS` + `metric_card_html()`            | `components.py`   |
| `CompletableFuture`            | Execução síncrona direta                    | `app.py`          |
| `RegressionService.java`       | `run_full_analysis()` + funções auxiliares  | `models.py`       |
| `EJML SimpleMatrix`            | `numpy.linalg.pinv()`                       | `models.py`       |
| Leitura de arquivos (Java I/O) | `parse_matrix()` / `parse_vector()`         | `models.py`       |
