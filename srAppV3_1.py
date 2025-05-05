import tkinter as visao  # Criando uma instância da biblioteca gráfica 
from tkinter import messagebox  # Biblioteca para exibir o tratamento de erros
import numpy as np  # Criando uma instância da biblioteca numpy para trabalhar com operações V/M
from numpy.linalg import inv, LinAlgError  # Importando a função que calcula a inversa de uma matriz

# Impressão dos arrays utilizando a biblioteca set_print
np.set_printoptions(suppress=True, precision=4)


def executarSR(c, M, b):
    print("\n|Iniciando Simplex Revisado|")
    print(f"Entrada c (Original): {c}")
    print(f"Entrada M (Original):\n{M}")
    print(f"Entrada b (Original): {b}")

 
    nr, nv = M.shape # nr: número de restrições, nv: número de variáveis
    print(f"Número de Restrições (nr): {nr}")
    print(f"Número de Variáveis Originais (nv): {nv}")

    # Estendendo a matriz M com a matriz identidade (variáveis de folga)
    M_ext = np.hstack((M, np.eye(nr)))
    print(f"\nM_ext (Matriz estendida com {nr} variáveis de folga):\n{M_ext}")

    # Extendendo o vetor de custos com zeros para as variáveis de folga
    c_ext = np.concatenate((c, np.zeros(nr)))
    print(f"\nc_ext (Vetor da função objetivo com 0s para folgas): {c_ext}")

    
    B_id = list(range(nv, nv + nr)) # Índices das variáveis básicas
    N_id = list(range(nv)) # Índices das variáveis não básicas 
    print(f"\nBase Inicial (B_id): {B_id}")
    print(f"Não Base Inicial (N_id): {N_id}")

    inteSimplex = 0 
   
    while True:
        inteSimplex += 1
        print(f"\n| Iteração {inteSimplex} |")
        print(f"Base Atual (B_id): {B_id}")
        print(f"Não Base Atual (N_id): {N_id}")

        # Pega a Matriz Original a partir das colunas de M_ext 
        B = M_ext[:, B_id]
        print(f"\nMatriz Base (B):\n{B}")

        # Inversa da Matriz Básica
        try:
            inv_B = inv(B)
            print(f"\nInversa da Matriz Base (inv_B):\n{inv_B}")
        except LinAlgError:
             print("\nErro: Não é possível calcular a inversa.")
             raise Exception("Não foi possível executar a operação!") 

        # Solução básica atual. xB = B_inversa * b
        # Para a solução básica ser viável, o xB deve ter valores não negativos
        xB = inv_B @ b
        print(f"\nValores das Variáveis Básicas (xB = inv_B @ b): {xB}")

        # Custos das variáveis básicas
        cB = c_ext[B_id]
        print(f"\nCoeficientes da Função Objetivo para Variáveis Básicas (cB): {cB}")

        # Custos das variáveis Duais. y = cB * B_inversa        
        y = cB @ inv_B
        print(f"\nCustos das variáveis Duais (y = cB @ inv_B): {y}")

       
        # Escolhe se a variável não básica entrar na base
        red_custos = c_ext[N_id] - y @ M_ext[:, N_id]
        print(f"\nCustos Reduzidos para as variaveis não Básicas (red_custos): {red_custos}")

       
        # Para MAX, a solução é ótima se todos os custos reduzidos forem <= 0 
        if all(red_custos <= 1e-8):
            print("\n| Condição Ótima Atingida |")
            # Cria o vetor incluindo folgas 
            x = np.zeros(nv + nr)
            for i, idx in enumerate(B_id):
                x[idx] = xB[i]
            print(f"\nVetor com (incluindo folgas): {x}")
            print(f"Solução Ótima (apenas variáveis originais): {x[:nv]}")
            print(f"Valor Ótimo da Função Objetivo: {c @ x[:nv]}")
            
            return x[:nv], c @ x[:nv], y, B_id, red_custos, M_ext, inv_B, c_ext

       
        
        # Se não encontrar a solução ótima, procura a variável não básica com o maior custo positivo
        sel_VB_idx_in_N = np.argmax(red_custos) # Posição na lista N_id
        sel_VB = N_id[sel_VB_idx_in_N]       # Índice real da variável em M_ext
        print(f"\nVariável que entra (índice em M_ext): {sel_VB} (Maior Custo Reduzido = {red_custos[sel_VB_idx_in_N]:.4f})")

        # d vai ser o vetor que mostra a direção depois da troca de uma variável
        d = inv_B @ M_ext[:, sel_VB]
        print(f"\nVetor 'd' (Coluna que entra no Tableau): {d}")

   
        # Verifica se o problema é ilimitado
   
        if all(d <= 1e-8):
            print("\n| Problema Ilimitado Detectado |")
            raise Exception("Problema ilimitado") # Interrompe a execução e sinaliza o erro

        # Verifica e calcula a variável que sai da base
        V_sb = []
        for i in range(nr):
             if d[i] > 1e-8: # Apenas se a variável básica diminuir ao aumentar
                 V_sb.append(xB[i] / d[i])
             else:
                 V_sb.append(np.inf) # Razão infinita se d[i] <= 0 
        V_sb = np.array(V_sb) 
        print(f"\nVetor de Razões (V_sb): {V_sb}")

        # Encontra o índice da menor razão 
        id_saida_idx_in_B = np.argmin(V_sb) 
        id_saida = B_id[id_saida_idx_in_B] 
        print(f"\nMaior variável (índice em M_ext): {id_saida} (Menor Razão = {V_sb[id_saida_idx_in_B]:.4f})")


        # Atualiza a base removendo a variável maior
        B_id[id_saida_idx_in_B] = sel_VB
        # Recalcula a lista de variáveis não básicas 
        N_id = [i for i in range(nv + nr) if i not in B_id]
        print(f"\nNova Base (B_id): {B_id}")
        print(f"Nova Não Base (N_id): {N_id}")
       



class SimplexApp:
   
    def __init__(self, root):
        self.root = root
        root.title("Simplex Revisado - Problema de Alocação de Perfil") 
        
      
        self.entrada_prefs = [] 
        self.entrada_restricoes = [] 
        self.tipos_restricoes = [] #

        
        visao.Label(root, text="Preferências dos Perfis - Disciplina: Pesquisa Operacional - Professor: Anand Subramanian", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=6)
        for i in range(2): 
            for j in range(3): 
                visao.Label(root, text=f"Perfil {i+1}, Programa {j+1}").grid(row=1 + i, column=j * 2)
                ent = visao.Entry(root, width=5)
                ent.grid(row=1 + i, column=j * 2 + 1)
                self.entrada_prefs.append(ent) 

       
        visao.Label(root, text="Tempo por Perfil").grid(row=3, column=0)
        self.tempo_perfil1 = visao.Entry(root, width=5)
        self.tempo_perfil1.grid(row=3, column=1)
        visao.Label(root, text="Perfil 1").grid(row=3, column=2)

        self.tempo_perfil2 = visao.Entry(root, width=5)
        self.tempo_perfil2.grid(row=3, column=3)
        visao.Label(root, text="Perfil 2").grid(row=3, column=4)

        
        visao.Label(root, text="Limite por Programa").grid(row=4, column=0, columnspan=6) 
        self.limites = []
        for i in range(3): 
            visao.Label(root, text=f"Programa {i+1}").grid(row=5, column=i * 2)
            ent = visao.Entry(root, width=5)
            ent.insert(0, "1") 
            ent.grid(row=5, column=i * 2 + 1)
            self.limites.append(ent) 

      
        self.btn_exec = visao.Button(root, text="Executar", command=self.executar_simplex)
        self.btn_exec.grid(row=6, column=0, columnspan=6, pady=10)

       
        self.output = visao.Text(root, height=20, width=100)
        self.output.grid(row=7, column=0, columnspan=6)

   
    def executar_simplex(self):
      
        self.output.delete(1.0, visao.END)
       

        try: 
            preferencias = np.array([float(ent.get()) for ent in self.entrada_prefs]).reshape(2, 3)
            c = np.array([preferencias[0, 0], preferencias[0, 1], preferencias[0, 2],
                          preferencias[1, 0], preferencias[1, 1], preferencias[1, 2]], dtype=float)
            self.nomes_variaveis = [f"X{i+1}" for i in range(6)] # Nomes para as variáveis originais.

            
            M = np.array([
            [1, 1, 1, 0, 0, 0],  # Tempo total Perfil 1
            [0, 0, 0, 1, 1, 1],  # Tempo total Perfil 2
            [1, 0, 0, 1, 0, 0],  #  1 pessoa pode ver Programa 1
            [0, 1, 0, 0, 1, 0],  #  1 pessoa pode ver Programa 2
            [0, 0, 1, 0, 0, 1]   #  1 pessoa pode ver Programa 3
            ], dtype=float)

            
            b = np.array([
                float(self.tempo_perfil1.get()), # Limite de tempo Perfil 1
                float(self.tempo_perfil2.get()), # Limite de tempo Perfil 2
                float(self.limites[0].get()),   # Limite Programa 1
                float(self.limites[1].get()),   # Limite Programa 2
                float(self.limites[2].get())    # Limite Programa 3
            ], dtype=float)

            print("\n|DADOS DE ENTRADA|")
            print(f"Vetor c (Função Objetivo): {c}")
            print(f"Matriz M (Restrições):\n{M}")
            print(f"Vetor b (Lado Direito):\n{b}")
            print("--------------------------------------")

          
            solucao, valor_otimo, dual, B_id, red_custos, M_ext, inv_B, c_ext = executarSR(c, M, b)

         
            self.output.insert(visao.END, f"\n| Resultados Finais |\n")
            self.output.insert(visao.END, f"Solução ótima (x_ij):\n{solucao}\n") 
            self.output.insert(visao.END, f"\nValor ótimo: {valor_otimo:.2f}\n") 

           
            recursos = ["1", "2", "3", "4", "5"] 
            recursos_info = [
                # Faz o tratamento numérico positivo com a função abs()
                f"{recursos[i]}: Dual = {dual[i]:.6f} - {'Escasso' if abs(dual[i]) > 1e-8 else 'Abundante'}"
                for i in range(len(b)) 
            ]
            self.output.insert(visao.END, "\nInformações dos Recursos (Custos Duais):\n")
            for r in recursos_info:
                self.output.insert(visao.END, f"{r}\n")

            
            ranges = self.calcular_ranges_objetivo(B_id, red_custos, M_ext, inv_B, c_ext, c)
            self.output.insert(visao.END, "\nINTERVALO DE VALORES QUE PODEMOS ALTERAR SEM MODIFICAR O PRIMAL:\n")
            self.output.insert(visao.END, "----------------------------------------------------------------------\n")
            self.output.insert(visao.END, "VARIÁVEL     COEFICIENTE      AUMENTO PERM.          REDUÇÃO PERM.\n")
            self.output.insert(visao.END, "----------------------------------------------------------------------\n")
           
            for var, current, increase, decrease in ranges:
                 
                 inc_str = f"{increase:.6f}" if np.isfinite(increase) else "INFINITO"
                 dec_str = f"{decrease:.6f}" if np.isfinite(decrease) else "INFINITO"
                 self.output.insert(visao.END, f"{var:<12} {current:<16.6f} {inc_str:<22} {dec_str:<22}\n")
            self.output.insert(visao.END, "----------------------------------------------------------------------\n")


       
        except Exception as e:
            
            self.output.insert(visao.END, f"\nAlgum problema na execução: {e}\n")
            messagebox.showerror("Erro na execução do Simplex!!", str(e))
            print(f"\nEsse é o tipo de erro: {e}") 


   
    def calcular_ranges_objetivo(self, B_id, red_custos, M_ext, inv_B, c_ext, c_original):
        nv = len(c_original) 
        # Pega os índices das variáveis básicas
        N_id = [i for i in range(M_ext.shape[1]) if i not in B_id]

        # Faz o cálculo dos duais 
        y = c_ext[B_id] @ inv_B

        ranges = [] # vetor para armazenar os intervalos 

        
        for var_index in range(nv):
            coef_atual = c_original[var_index] # Coeficiente atual da Função Objetivo
            aument_permit = float('inf') # Inicializa aumento permitido 
            redu_permit = float('inf') # Inicializa redução permitida 

            # Implementação dos intervalos 
            if var_index in B_id:
                basic_pos = B_id.index(var_index) # Posição na base

                # Interação sobre as variáveis não básicas para encontrar limites
                for i, nb_index in enumerate(N_id):
                    # cVNB: Coluna da variável não básica no tableau ótimo (B_inversa * M_ext_coluna)
                    cVNB = inv_B @ M_ext[:, nb_index]

                    # Se o elemento cVNB na posição da variável básica é positivo, ele limita o AUMENTO
                    if cVNB[basic_pos] > 1e-8:
                        # Aumento permitido = -custo_reduzido_não_básica / cVNB[posição_básica]
                        aument_permit = min(aument_permit, -red_custos[i] / cVNB[basic_pos])
                    # Se o elemento cVNB na posição da variável básica é negativo, ele limita a REDUÇÃO
                    elif cVNB[basic_pos] < -1e-8:
                        
                        redu_permit = min(redu_permit, -red_custos[i] / cVNB[basic_pos])


            else:
                # Encontrar a posição da variável não básica
                try:
                    nb_pos_in_N = N_id.index(var_index)
                except ValueError: # Retorna um erro se não existir um índex
                     print(f"\n Variável {var_index} não encontrada em N_id!")
                     continue # Depois ele pula esta variável

               
                original_red_cost_j = red_custos[nb_pos_in_N]

                # permite AUMENTO no coeficiente
                if original_red_cost_j < -1e-8:
                      # Se o custo reduzido for positivo, permite REDUÇÃO no coeficiente
                     aument_permit = -original_red_cost_j
               
                elif original_red_cost_j > 1e-8:
                     
                     redu_permit = -original_red_cost_j # Coloca o custo negativo quando for positivo


            # Adiciona os intervalos calculados
            ranges.append((self.nomes_variaveis[var_index], coef_atual, aument_permit, redu_permit))

        return ranges # Retorna a lista de intervalos 


if __name__ == "__main__":
    root = visao.Tk() 
    app = SimplexApp(root) 
    root.mainloop() 