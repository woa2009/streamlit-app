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
echo   EXECUTANDO STREAMLIT...
echo ======================================================

:: Executa o app
streamlit run app.py

pause