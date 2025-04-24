import tkinter as visao  # Criando uma instância da biblioteca gráfica 
from tkinter import messagebox  # Biblioteca para exibir o tratamento de erros
import numpy as np  # Criando uma instância da biblioteca numpy para trabalhar com operações V/M
from numpy.linalg import inv  # Importando a função que calcula a inversa de uma matriz


def executarSR(c, M, b):
    nr, nv = M.shape  # nr: número de restrições, nv: número de variáveis

    # Estendendo a matriz M com a matriz identidade (variáveis de folga)
    M_ext = np.hstack((M, np.eye(nr)))
    print("\n[Matriz estendida M_ext]:")
    print(M_ext)

    # Extendendo o vetor de custos com zeros para as variáveis de folga
    c_ext = np.concatenate((c, np.zeros(nr)))
    print("\n[Vetor de custos estendido c_ext]:")
    print(c_ext)

    # Índices das variáveis básicas (inicialmente as folgas)
    B_id = list(range(nv, nv + nr))

    # Índices das variáveis não básicas (variáveis de decisão originais)
    N_id = list(range(nv))

    while True:
        # Matriz básica
        B = M_ext[:, B_id]
        print("\n[Matriz básica B]:")
        print(B)

        # Inversa da matriz básica
        inversa_B = inv(B)
        print("\n[Inversa da matriz básica inversa_B]:")
        print(inversa_B)

        # Solução básica atual
        xB = inversa_B @ b
        print("\n[Solução básica xB = inversa_B @ b]:")
        print(xB)

        # Custos das variáveis básicas
        cB = c_ext[B_id]
        print("\n[Custos das variáveis básicas cB]:")
        print(cB)

        # Vetor de multiplicadores simplex (solução dual)
        y = cB @ inversa_B
        print("\n[Vetor de multiplicadores simplex y = cB @ inversa_B]:")
        print(y)

        # Custos reduzidos das variáveis não básicas
        red_custos = c_ext[N_id] - y @ M_ext[:, N_id]
        print("\n[Custos reduzidos]:")
        print(red_custos)

        # Verifica otimalidade (todas as reduções de custo não são positivas)
        if all(red_custos <= 1e-8):
            x = np.zeros(nv + nr)
            for i, idx in enumerate(B_id):
                x[idx] = xB[i]
            print("\n[Solução ótima encontrada]")
            return x[:nv], c @ x[:nv], y  # Retorna solução primal, valor ótimo, solução dual

        # Seleciona a variável que entra na base (maior custo reduzido)
        sel_VB = N_id[np.argmax(red_custos)]
        print(f"\n[Variável que entra na base: {sel_VB}]")

        # d vai ser o vetor que mostra a direção depois da troca de uma variável
        d = inversa_B @ M_ext[:, sel_VB]
        print("\n[Direção d = inversa_B @ coluna selecionada]:")
        print(d)

        # Verifica se o problema é ilimitado (nenhuma direção positiva)
        if all(d <= 1e-8):
            raise Exception("Problema ilimitado")

        # Verifica e calcula a variável que sai da base
        V_sb = [xB[i] / d[i] if d[i] > 0 else np.inf for i in range(nr)]
        id_saida = np.argmin(V_sb)
        print(f"[Variável que sai da base: índice {id_saida} (posição na base)]")

        # Atualiza a base
        B_id[id_saida] = sel_VB
        N_id = [i for i in range(nv + nr) if i not in B_id]


class SimplexApp:
    def __init__(self, master):
        self.master = master
        master.title("Simplex Revisado - Problema de Alocação de Perfil")  
        
        self.label = visao.Label(master, text="Disciplina: Pesquisa Operacional - Professor: Anand Subramanian")
        self.label.pack(pady=10)

        self.run_button = visao.Button(master, text="Executar", command=self.executar_simplex)
        self.run_button.pack(pady=10)

        self.output = visao.Text(master, height=15, width=80)
        self.output.pack(pady=10)

    def executar_simplex(self):
        # Matriz das preferências dos usuários
        preferencias = np.array([
            [5, 7, 2],  # Perfil 1
            [8, 4, 3]   # Perfil 2
        ])

        # Função objetivo: satisfação total de cada perfil para cada programa
        c = np.array([
            preferencias[0, 0], preferencias[0, 1], preferencias[0, 2],
            preferencias[1, 0], preferencias[1, 1], preferencias[1, 2]
        ], dtype=float)

        # Restrições do problema
        M = np.array([
            [1, 1, 1, 0, 0, 0],  # Tempo total permitido para Perfil 1
            [0, 0, 0, 1, 1, 1],  # Tempo total permitido para Perfil 2
            [1, 0, 0, 1, 0, 0],  # Limite: apenas 1 pessoa pode ver Programa 1
            [0, 1, 0, 0, 1, 0],  # Limite: apenas 1 pessoa pode ver Programa 2
            [0, 0, 1, 0, 0, 1]   # Limite: apenas 1 pessoa pode ver Programa 3
        ], dtype=float)

        # Recursos disponíveis (tempo e limites por programa)
        b = np.array([2, 2, 1, 1, 1], dtype=float)

        try:
            solucao, valor_otimo, dual = executarSR(c, M, b)

            recursos = ["1", "2", "3", "4", "5"]
            recursos_info = [
                f"{recursos[i]}: Dual = {dual[i]:.2f} - {'Escasso' if abs(dual[i]) > 1e-6 else 'Abundante'}"
                for i in range(len(b))
            ]

            self.output.delete(1.0, visao.END)
            self.output.insert(visao.END, f"Solução ótima (x_ij):\n{solucao}\n")
            self.output.insert(visao.END, f"\nValor ótimo: {valor_otimo:.2f}\n")
            self.output.insert(visao.END, "\nAnálise dos Recursos:\n")
            for r in recursos_info:
                self.output.insert(visao.END, f"{r}\n")

        except Exception as e:
            messagebox.showerror("Erro", str(e))


if __name__ == "__main__":
    root = visao.Tk()  
    app = SimplexApp(root) 
    root.mainloop()
