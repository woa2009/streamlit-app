@echo off
echo ============================================
echo  Verificando ambiente Python em uso...
echo ============================================

where python
where streamlit
python --version

echo.
echo ============================================
echo  Atualizando openpyxl no ambiente correto...
echo ============================================

python -m pip install --upgrade pip
python -m pip install "openpyxl>=3.1.0" --upgrade --force-reinstall
python -m pip install -r requirements.txt --upgrade

echo.
echo ============================================
echo  Versao instalada apos atualizacao:
echo ============================================

python -m pip show openpyxl

echo.
echo ============================================
echo  Iniciando o Streamlit...
echo ============================================

python -m streamlit run app.py

pause
