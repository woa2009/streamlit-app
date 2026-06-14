import numpy as np
import os
import time


def gerar_matriz_txt(nome_arquivo, num_linhas, num_colunas, bloco_tamanho=5000):
    """
    Gera um arquivo .txt contendo uma matriz de dados espectrais simulados.
    Salva em blocos para não estourar a RAM.
    """
    print(f"Criando '{nome_arquivo}' ({num_linhas} linhas x {num_colunas} colunas)...")
    inicio = time.time()

    with open(nome_arquivo, 'w') as f:
        linhas_restantes = num_linhas

        while linhas_restantes > 0:
            linhas_atual = min(bloco_tamanho, linhas_restantes)

            dados_bloco = np.random.uniform(0.0, 10.0, size=(linhas_atual, num_colunas))
            np.savetxt(f, dados_bloco, fmt='%.5f', delimiter='\t')

            linhas_restantes -= linhas_atual

    fim = time.time()
    tamanho_mb = os.path.getsize(nome_arquivo) / (1024 * 1024)
    print(f"  -> Finalizado em {fim - inicio:.2f}s | Tamanho: {tamanho_mb:.2f} MB")


# --- CONFIGURAÇÃO DAS DIMENSÕES ---
NUM_COLUNAS_X = 723   # variáveis espectrais
LINHAS_CAL    = 30165 # amostras de calibração (~1/3)
LINHAS_TESTE  = 60330 # amostras de teste (~2/3)

print("--- INICIANDO A GERAÇÃO DOS DADOS SIMULADOS ---\n")

# 1. Xcal: matriz de calibração (30165 x 723)
gerar_matriz_txt("Xcal.txt", LINHAS_CAL, NUM_COLUNAS_X)

# 2. Ycal: vetor de calibração (30165 x 1)
gerar_matriz_txt("Ycal.txt", LINHAS_CAL, 1)

# 3. Xteste: matriz de teste (60330 x 723)
gerar_matriz_txt("Xteste.txt", LINHAS_TESTE, NUM_COLUNAS_X)

# 4. Yteste: vetor de teste (60330 x 1)
gerar_matriz_txt("Yteste.txt", LINHAS_TESTE, 1)

print("\nProcesso concluído! Arquivos gerados: Xcal.txt, Ycal.txt, Xteste.txt, Yteste.txt")
