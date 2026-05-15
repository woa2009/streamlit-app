# ⚗️ Regressão Espectral — Streamlit

Conversão do projeto **Vaadin Flow (Java)** para **Streamlit (Python)**.

Implementa RLM Simples, RLM + Bagging e RLM + Subagging com otimização
de hiperparâmetros por validação cruzada 5-fold.

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

| Arquivo               | Tipo   | Regras |
|-----------------------|--------|--------|
| Xcal `.txt/csv/xlsx`  | Matriz | Colunas por espaço (`.txt`), vírgula ou `;` (`.csv`), planilha sem cabeçalho (`.xlsx`) |
| Ycal `.txt/csv/xlsx`  | Vetor  | Um valor por linha; primeira coluna em `.csv`/`.xlsx` |
| Xteste `.txt/csv/xlsx`| Matriz | Igual ao Xcal |
| Yteste `.txt/csv/xlsx`| Vetor  | Igual ao Ycal |

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

| Vaadin (Java)              | Streamlit (Python)              |
|----------------------------|---------------------------------|
| `@Route("")`               | `st.set_page_config()`          |
| `Tabs`                     | `st.tabs()`                     |
| `Upload + MemoryBuffer`    | `st.file_uploader()`            |
| `TextField`                | `st.text_input()`               |
| `Button + ClickListener`   | `st.button()`                   |
| `ProgressBar`              | `st.progress()`                 |
| `Grid<Entidade>`           | `st.dataframe()`                |
| `Notification`             | `st.success()` / `st.error()`   |
| SVG inline (gráficos)      | `plotly` interativo             |
| `CompletableFuture`        | Execução síncrona direta        |
| `RegressionService` (Java) | Funções Python com NumPy        |
| `EJML SimpleMatrix`        | `numpy.linalg.pinv()`           |
