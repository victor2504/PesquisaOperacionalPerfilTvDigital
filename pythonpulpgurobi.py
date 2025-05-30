import shutil
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary, value, GUROBI_CMD # Importamos GUROBI_CMD
import time

def resolver_problema_com_pulp_gurobi(): 
    print("\n--- Solução com Gurobi ---") 
    start = time.time()

    # Definir o modelo de otimização
    model = LpProblem("Alocacao_Perfis_Programas", LpMaximize)

   
    x = [LpVariable(f"x{i+1}", cat=LpBinary) for i in range(6)]

    # Função Objetivo: max 8 x1 + 4 x2 + 2 x3 + 7 x4 + 3 x5 + 5 x6
    model += (8 * x[0] + 4 * x[1] + 2 * x[2] +
              7 * x[3] + 3 * x[4] + 5 * x[5]), "Total_Preferencias"

  

    # Restrição 1: Perfil 1 pode assistir no máximo 2 programas
    # (Corresponde a x1 + x2 + x3 <= 2)
    model += x[0] + x[1] + x[2] <= 2, "Perfil_1_Max_Programas"

    # Restrição 2: Perfil 2 pode assistir no máximo 2 programas
    # (Corresponde a x4 + x5 + x6 <= 2)
    model += x[3] + x[4] + x[5] <= 2, "Perfil_2_Max_Programas"

    # Restrição 3: Programa 1 pode ser assistido por no máximo 1 perfil
    # (Corresponde a x1 + x4 <= 1)
    model += x[0] + x[3] <= 1, "Programa_1_Max_Perfis"

    # Restrição 4: Programa 2 pode ser assistido por no máximo 1 perfil
    # (Corresponde a x2 + x5 <= 1)
    model += x[1] + x[4] <= 1, "Programa_2_Max_Perfis"

    # Restrição 5: Programa 3 pode ser assistido por no máximo 1 perfil
    # (Corresponde a x3 + x6 <= 1)
    model += x[2] + x[5] <= 1, "Programa_3_Max_Perfis"

   
    gurobi_executables = ["gurobi_cl", "gurobi.sh", "gurobi.bat"]
    gurobi_found = False
    for exec_name in gurobi_executables:
        if shutil.which(exec_name):
            gurobi_found = True
            break

    if gurobi_found:
        # msg=True para exibir o log do solver Gurobi
        # options=[] pode ser usado para passar opções específicas do Gurobi
        solver = GUROBI_CMD(msg=True, options=[])
        model.solve(solver)
    else:
        print("\nErro: Nenhum executável do Gurobi ('gurobi_cl', 'gurobi.sh', 'gurobi.bat') foi encontrado no PATH do sistema.")
        print("Certifique-se de que o Gurobi está instalado e acessível.")
        return # Sai da função se Gurobi não estiver disponível

    end = time.time()

    print(f"\n---Solução (Gurobi) ---") 
    print(f"Status da Solução: {model.status}")
    print(f"Tempo total de execução: {end - start:.4f} segundos")
    print("Valor ótimo da função objetivo:", value(model.objective))

    print("\nAtribuições de Perfil para Programa:")
    # Mapeamento para traduzir o índice da variável de volta para (perfil, programa)
    var_idx_to_profile_program = {
        0: (1, 1), 1: (1, 2), 2: (1, 3),
        3: (2, 1), 4: (2, 2), 5: (2, 3)
    }

    # Dicionário para armazenar o que cada perfil assiste
    atribuicoes_por_perfil = {1: [], 2: []}

    for i in range(6):
        if x[i].value() is not None and x[i].value() >= 0.99:
            perfil, programa = var_idx_to_profile_program[i]
            atribuicoes_por_perfil[perfil].append(programa)

    for perfil, programas_assistidos in atribuicoes_por_perfil.items():
        if programas_assistidos:
            print(f"Perfil {perfil} assiste ao(s) programa(s): {', '.join(map(str, programas_assistidos))}")
        else:
            print(f"Perfil {perfil} não assiste a nenhum programa.")
    print("---------------------------------------")


resolver_problema_com_pulp_gurobi() 
