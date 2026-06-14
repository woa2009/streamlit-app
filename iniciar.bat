@echo off
title Streamlit - Regressao Espectral
color 0A
echo ======================================================
echo   INICIANDO AMBIENTE ANACONDA...
echo ======================================================
:: Ativa o Anaconda3
call C:\Users\wagner.araujo\anaconda3\Scripts\activate.bat C:\Users\wagner.araujo\anaconda3

echo ======================================================
echo   ACESSANDO DIRETORIO DO PROJETO...
echo ======================================================
:: Vai para o diretorio do projeto
cd /d L:\TCCI\D_Programacao\streamlit-app
if errorlevel 1 (
    echo ERRO: Diretorio nao encontrado!
    echo Verifique se o disco L: esta conectado.
    pause
    exit /b 1
)

echo ======================================================
echo   CORRIGINDO DEPENDENCIAS (NumPy + openpyxl)...
echo ======================================================
:: Rebaixa NumPy para versao 1.x (compativel com pandas/numexpr/bottleneck)
python -m pip install "numpy<2" --force-reinstall -q
:: Atualiza openpyxl
python -m pip install "openpyxl>=3.1.0" --upgrade --force-reinstall -q

echo ======================================================
echo   EXECUTANDO STREAMLIT (porta 8502)...
echo ======================================================
:: Executa o app na porta 8502 (8501 reservada pelo sistema Windows)
streamlit run app.py --server.port 8502
pause
