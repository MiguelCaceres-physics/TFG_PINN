from graphviz import Digraph

def add_compressed_layer(dot, name, total_neurons, label):
    """
    Capa comprimida: 1 neurona arriba, ⋮, 2 neuronas abajo.
    Devuelve lista de neuronas representativas (sin los puntos).
    """
    with dot.subgraph(name=f"cluster_{name}") as c:
        c.attr(label=f"{label} ({total_neurons})", color="gray70")
        c.attr("node", shape="circle", fixedsize="true", width="0.38")

        top = f"{name}_top"
        mid = f"{name}_mid"
        dots = f"{name}_dots"
        bot1 = f"{name}_bot1"
        bot2 = f"{name}_bot2"

        c.node(top, "")
        c.node(mid, "")
        c.node(dots, "⋮", shape="plaintext")
        c.node(bot1, "")
        c.node(bot2, "")

        # Para que queden en columna (rank=same dentro del cluster ayuda, pero esto lo refuerza)
        c.attr(rankdir="TB")

    return [top, mid, bot1, bot2]


def fully_connect(dot, left_nodes, right_nodes, **edge_attrs):
    """Conecta todas las neuronas representativas de izquierda a derecha."""
    for a in left_nodes:
        for b in right_nodes:
            dot.edge(a, b, **edge_attrs)


dot = Digraph("PINN_FC_3x50", format="png")
dot.attr(rankdir="LR", splines="line", nodesep="0.6", ranksep="1.1")  # splines=line => rectas
dot.attr("edge", penwidth="1")  # grosor estándar

# Entrada: t (tiempo normalizado)
dot.node("t", "t\n[0,1]", shape="circle", fixedsize="true", width="0.5")

# Capas ocultas: 3x50
H1 = add_compressed_layer(dot, "H1", 50, "Capa oculta 1")
H2 = add_compressed_layer(dot, "H2", 50, "Capa oculta 2")
H3 = add_compressed_layer(dot, "H3", 50, "Capa oculta 3")

# Salida: x_hat (desplazamiento predicho)
dot.node("xhat", "x̂", shape="circle", fixedsize="true", width="0.5")

# Conexiones: “denso” representado con conexiones completas entre nodos representativos
# Entrada -> H1 (conecta t a todas las representativas de H1)
for n in H1:
    dot.edge("t", n)

# H1 -> H2, H2 -> H3, H3 -> Salida
fully_connect(dot, H1, H2)
fully_connect(dot, H2, H3)

for n in H3:
    dot.edge(n, "xhat")

dot.render(view=True)