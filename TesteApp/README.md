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

## ☁️ Deploy no Streamlit Community Cloud (gratuito)

1. Suba este repositório no GitHub
2. Acesse https://share.streamlit.io
3. Clique em **New app** → selecione o repositório e o arquivo `app.py`
4. Clique em **Deploy**

## 📁 Formato dos arquivos de entrada

| Arquivo  | Formato                                      |
|----------|----------------------------------------------|
| Xcal.txt | Matriz: uma amostra por linha, colunas separadas por espaço |
| Ycal.txt | Vetor: um valor por linha                    |
| Xteste.txt | Igual ao Xcal                              |
| Yteste.txt | Igual ao Ycal                              |

## 🔄 Correspondência Vaadin → Streamlit

| Vaadin (Java)              | Streamlit (Python)         |
|----------------------------|----------------------------|
| `@Route("")`               | `st.set_page_config()`     |
| `Tabs`                     | `st.tabs()`                |
| `Upload + MemoryBuffer`    | `st.file_uploader()`       |
| `TextField`                | `st.text_input()`          |
| `Button + ClickListener`   | `st.button()`              |
| `ProgressBar`              | `st.progress()`            |
| `Grid<Entidade>`           | `st.dataframe()`           |
| `Notification`             | `st.success()` / `st.error()` |
| SVG inline (gráficos)      | `plotly` interativo        |
| `CompletableFuture`        | Execução síncrona direta   |
| `RegressionService` (Java) | Funções Python com NumPy   |
| `EJML SimpleMatrix`        | `numpy.linalg.pinv()`      |
