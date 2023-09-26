import matplotlib.pyplot as plt


def plotResultsStocks(resultsList):
    fig, ax = plt.subplots()
    # Configurar el alto de la figura (Valor = 5), ancho de la figura (valor = 10), color ('lightgrey') y título ('Lluvia media mensual')
    fig.suptitle('Predicción de valores Nasdaq', fontsize=20)
    fig.set_facecolor('lightgrey')
    fig.set_figheight(5)
    fig.set_figwidth(10)
    # Etiquetas de los ejes x, y
    fontX = {'family': 'serif', 'color': 'darkblue', 'size': 10}
    fontY = {'family': 'serif', 'color': 'darkred', 'size': 10}
    ax.set_xlabel('Meses', fontX)
    ax.set_ylabel('Valores', fontY)
    # Etiquetas de las reglas
    ax.tick_params(axis='x', labelsize=14, labelrotation=90)
    ax.tick_params(axis='y', labelsize=13)
    # Grilla horizontal, color gris, linea de puntos, espesor 1.
    ax.grid(axis='y', color='grey', linestyle=':', linewidth=1)

    for list in resultsList:
        ax.plot(list, linestyle='--', linewidth=2, marker='o', markersize=5)

    ax.legend(['Valores reales', 'Predicción'], loc='upper center')

    plt.show()


def plotResultsTrades(resultsList):
    fig, ax = plt.subplots()
    # Configurar el alto de la figura (Valor = 5), ancho de la figura (valor = 10), color ('lightgrey') y título ('Lluvia media mensual')
    fig.suptitle('Performance en trades diarios', fontsize=20)
    fig.set_facecolor('lightgrey')
    fig.set_figheight(5)
    fig.set_figwidth(10)
    # Etiquetas de los ejes x, y
    fontX = {'family': 'serif', 'color': 'darkblue', 'size': 10}
    fontY = {'family': 'serif', 'color': 'darkred', 'size': 10}
    ax.set_xlabel('Meses', fontX)
    ax.set_ylabel('Valores', fontY)
    # Etiquetas de las reglas
    ax.tick_params(axis='x', labelsize=14, labelrotation=90)
    ax.tick_params(axis='y', labelsize=13)
    # Grilla horizontal, color gris, linea de puntos, espesor 1.
    ax.grid(axis='y', color='grey', linestyle=':', linewidth=1)

    for list in resultsList:
        ax.plot(list, linestyle='--', linewidth=2, marker='o', markersize=5)

    plt.show()

def plotResultsLoss(resultsList):
    fig, ax = plt.subplots()
    # Configurar el alto de la figura (Valor = 5), ancho de la figura (valor = 10), color ('lightgrey') y título ('Lluvia media mensual')
    fig.suptitle('Loss', fontsize=20)
    fig.set_facecolor('lightgrey')
    fig.set_figheight(5)
    fig.set_figwidth(10)
    # Etiquetas de los ejes x, y
    fontX = {'family': 'serif', 'color': 'darkblue', 'size': 10}
    fontY = {'family': 'serif', 'color': 'darkred', 'size': 10}
    ax.set_xlabel('Meses', fontX)
    ax.set_ylabel('Valores', fontY)
    # Etiquetas de las reglas
    ax.tick_params(axis='x', labelsize=14, labelrotation=90)
    ax.tick_params(axis='y', labelsize=13)
    # Grilla horizontal, color gris, linea de puntos, espesor 1.
    ax.grid(axis='y', color='grey', linestyle=':', linewidth=1)

    for list in resultsList:
        ax.plot(list, linestyle='--', linewidth=2, marker='o', markersize=5)

    plt.show()