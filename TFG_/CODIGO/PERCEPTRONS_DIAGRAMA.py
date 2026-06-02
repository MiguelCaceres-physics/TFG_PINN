import matplotlib
# Forzamos un backend que no dependa de una ventana interactiva
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import schemdraw
import schemdraw.elements as elm

# Usamos show=False para evitar que intente llamar a 'display' de Jupyter
with schemdraw.Drawing(show=False) as d:
    d.config(fontsize=11)

    # Función auxiliar para crear círculos de forma genérica y evitar el AttributeError
    def draw_circle(x, y, label, fill, color):
        # En 0.22, 'Cnf' o 'Element' con shape son los más estables
        node = d.add(elm.Encircle(radius=0.5).at((x, y)).fill(fill).color(color))
        d.add(elm.Text(label).at((x, y)))
        return node

    # --- ENTRADAS ---
    x1 = draw_circle(0, 0, '$x_1$', '#D5EAD8', '#53A65B')
    x2 = draw_circle(0, -1.8, '$x_2$', '#D5EAD8', '#53A65B')
    xn = draw_circle(0, -3.6, '$x_n$', '#D5EAD8', '#53A65B')
    bias = draw_circle(0, -5.4, '1', '#D5EAD8', '#53A65B')

    # Etiquetas laterales
    d.add(elm.Label(label='Input 1: $x_1$', loc='left').at(x1.west))
    d.add(elm.Label(label='Input n: $x_n$', loc='left').at(xn.west))
    d.add(elm.Label(label='Bias (1)', loc='left').at(bias.west))

    # --- NODO SUMA ---
    # Usamos Encircle de nuevo pero más grande
    suma = d.add(elm.Encircle(radius=0.8).at((4, -2.7)).fill('#D1E4F9').color('#4A90E2'))
    d.add(elm.Text(r'$\Sigma$', fontsize=22).at((4, -2.7)))
    d.add(elm.Label(label='Summing\nJunction', loc='bottom').at(suma.south))

    # --- PESOS Y LÍNEAS ---
    inputs = [x1, x2, xn, bias]
    w_labels = ['$w_1$', '$w_2$', '$w_n$', '$w_0$']
    for i, lbl in enumerate(w_labels):
        d.add(elm.Line().at(inputs[i].east).to((4, -2.7)).color('#2C5E77').zorder(0))
        # Nodo de peso intermedio
        mid_x, mid_y = (0 + 4)/2, (inputs[i].center[1] + -2.7)/2
        w_node = d.add(elm.Encircle(radius=0.25).at((mid_x, mid_y)).fill('#2C5E77').color('#2C5E77'))
        d.add(elm.Text(lbl, color='white', fontsize=8).at((mid_x, mid_y)))

    # --- ACTIVACIÓN ---
    # En lugar de Rect que da problemas, usamos una caja (Data) o un nodo cuadrado
    act_box = d.add(elm.Rect(w=1.4, h=1.4).at((6.5, -3.4)).fill('#D1E4F9').color('#4A90E2'))
    # Línea de la sigmoide
    d.add(elm.Line().at((6.8, -3.1)).to((7.6, -2.3)).color('#1C3D67'))
    d.add(elm.Label(label=r'Activation $\sigma(z)$', loc='top').at((7.2, -2.0)))

    # --- CONEXIONES ---
    d.add(elm.Line(arrow='>').at(suma.east).to((6.5, -2.7)))
    
    out = d.add(elm.Encircle(radius=0.6).at((10, -2.7)).fill('#F5DCCF').color('#D97E3F'))
    d.add(elm.Line(arrow='>').at((7.9, -2.7)).to(out.west))
    d.add(elm.Text(r'$\hat{y}$').at((10, -2.7)))
    
    d.add(elm.Line(arrow='>').at(out.east).length(1)).label('$y$\n(Target)', loc='right')

    # GUARDADO DIRECTO
    d.save('resultado_perceptron.png')

print("¡Éxito! El diagrama se ha guardado como 'resultado_perceptron.png'")