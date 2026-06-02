import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Configuración General

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo:", device)

# Archivo HFM
mat_file = "Cylinder2D.mat"

# Número de puntos de entrenamiento
num_train = 1000

# Nivel de ruido aplicado a u y v
ruido = 0.3

# Guardado en Google Drive
from google.colab import drive
drive.mount('/content/drive')

output_dir = f"/content/drive/MyDrive/TFG_HFM/resultados_N{num_train}_ruido{int(ruido*100)}"
os.makedirs(output_dir, exist_ok=True)
# Entrenamiento
adam_epochs = 20000
adam_lr = 1e-3
lbfgs_max_iter = 500

# Pesos de pérdidas
w_phys = 1.0
w_data = 10.0

# Tiempo máximo usado del dataset
# Mantengo t <= 7 porque era tu configuración original.
t_max_train = 7.0

# CARGAMOS EL DATA SET

data_mat = loadmat(mat_file)

U_star = data_mat["U_star"]
V_star = data_mat["V_star"]
P_star = data_mat["P_star"]
t_star = data_mat["t_star"]
X_star = data_mat["x_star"]
Y_star = data_mat["y_star"]

N = X_star.shape[0]
Nt = t_star.shape[0]

XX = np.tile(X_star, (1, Nt))
YY = np.tile(Y_star, (1, Nt))
TT = np.tile(t_star.T, (N, 1))

x = XX.flatten()[:, None]
y = YY.flatten()[:, None]
t = TT.flatten()[:, None]

u = U_star.flatten()[:, None]
v = V_star.flatten()[:, None]
p = P_star.flatten()[:, None]

data_all = np.concatenate([x, y, t, u, v, p], axis=1)

# Usamos solo el corte temporal t <= 7, para no sobre cargar el trabajo.
mask = data_all[:, 2] <= t_max_train
data_domain = data_all[mask]

print("Puntos disponibles en HFM tras filtro temporal:", data_domain.shape[0])
print(f"Rango x: [{data_domain[:,0].min():.3f}, {data_domain[:,0].max():.3f}]")
print(f"Rango y: [{data_domain[:,1].min():.3f}, {data_domain[:,1].max():.3f}]")
print(f"Rango t: [{data_domain[:,2].min():.3f}, {data_domain[:,2].max():.3f}]")

# Muestreo de puntos de entrenamiento
num_train = min(num_train, data_domain.shape[0])
idx = np.random.choice(data_domain.shape[0], num_train, replace=False)

x_train = data_domain[idx, 0:1]
y_train = data_domain[idx, 1:2]
t_train = data_domain[idx, 2:3]

u_train = data_domain[idx, 3:4]
v_train = data_domain[idx, 4:5]
p_train = data_domain[idx, 5:6]  # SOLO VALIDACIÓN, NO ENTRENAMIENTO

X_train = np.hstack([x_train, y_train, t_train])

print("\nPuntos de entrenamiento:", num_train)
print(f"Rango u train: [{u_train.min():.4f}, {u_train.max():.4f}]")
print(f"Rango v train: [{v_train.min():.4f}, {v_train.max():.4f}]")
print(f"Rango p train: [{p_train.min():.4f}, {p_train.max():.4f}]")

# Ruido
u_std = np.std(u_train)
v_std = np.std(v_train)

u_train_noisy = u_train + ruido * u_std * np.random.randn(*u_train.shape)
v_train_noisy = v_train + ruido * v_std * np.random.randn(*v_train.shape)
# Comprobación ruido
ruido_rel_u = np.linalg.norm(u_train_noisy - u_train) / np.linalg.norm(u_train)
ruido_rel_v = np.linalg.norm(v_train_noisy - v_train) / np.linalg.norm(v_train)

print(f"Ruido configurado: {ruido*100:.1f}%")
print(f"Ruido relativo real en u: {ruido_rel_u:.4e}")
print(f"Ruido relativo real en v: {ruido_rel_v:.4e}")
print("Diferencia máxima en u:", np.max(np.abs(u_train_noisy - u_train)))
print("Diferencia máxima en v:", np.max(np.abs(v_train_noisy - v_train)))
# Tensores
X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
u_train_t = torch.tensor(u_train_noisy, dtype=torch.float32, device=device)
v_train_t = torch.tensor(v_train_noisy, dtype=torch.float32, device=device)

# GENERAMOS EL MODELO PINN

class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()

        net = []
        for i in range(len(layers) - 2):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            net.append(nn.Tanh())

        net.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net)

        self.init_weights()

    def init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# Entrada: x, y, t
# Salida: u, v, p
model = PINN([3, 64, 64, 64, 64, 64, 64, 3]).to(device)

# Parámetros físicos identificables
# lambda1 esperado ≈ 1
# lambda2 esperado ≈ 0.01, Re ≈ 100
log_l1 = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=device))
log_l2 = nn.Parameter(torch.tensor(math.log(0.01), dtype=torch.float32, device=device))

# GRADIENTE Y RESIDUOS

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]


def pinn_loss(X_data, u_data, v_data):

    X = X_data.clone().detach().requires_grad_(True)

    pred = model(X)

    u_pred = pred[:, 0:1]
    v_pred = pred[:, 1:2]
    p_pred = pred[:, 2:3]

    lambda1 = torch.exp(log_l1)
    lambda2 = torch.exp(log_l2)

    # Derivadas de u
    grads_u = grad(u_pred, X)
    u_x = grads_u[:, 0:1]
    u_y = grads_u[:, 1:2]
    u_t = grads_u[:, 2:3]

    u_xx = grad(u_x, X)[:, 0:1]
    u_yy = grad(u_y, X)[:, 1:2]

    # Derivadas de v
    grads_v = grad(v_pred, X)
    v_x = grads_v[:, 0:1]
    v_y = grads_v[:, 1:2]
    v_t = grads_v[:, 2:3]

    v_xx = grad(v_x, X)[:, 0:1]
    v_yy = grad(v_y, X)[:, 1:2]

    # Derivadas de p
    grads_p = grad(p_pred, X)
    p_x = grads_p[:, 0:1]
    p_y = grads_p[:, 1:2]

    # Residuos físicos
    continuidad = u_x + v_y

    momento_x = (u_t+ lambda1 * (u_pred * u_x + v_pred * u_y)+ p_x- lambda2 * (u_xx + u_yy))

    momento_y = (v_t+ lambda1 * (u_pred * v_x + v_pred * v_y)+ p_y- lambda2 * (v_xx + v_yy))

    # Pérdida de datos: solo velocidades
    loss_u = torch.mean((u_pred - u_data) ** 2)
    loss_v = torch.mean((v_pred - v_data) ** 2)

    # Pérdida física
    loss_cont = torch.mean(continuidad ** 2)
    loss_mom_x = torch.mean(momento_x ** 2)
    loss_mom_y = torch.mean(momento_y ** 2)

    loss_data = loss_u + loss_v
    loss_phys = loss_cont + loss_mom_x + loss_mom_y

    loss_total = w_data * loss_data + w_phys * loss_phys #  el peso determina la importancia del modelo a la pérdida o incidencia que queramos desear

    return loss_total, loss_data, loss_phys, loss_u, loss_v, loss_cont, loss_mom_x, loss_mom_y


# Entrenamiento por ADAM

params = list(model.parameters()) + [log_l1, log_l2]
optimizer = torch.optim.Adam(params, lr=adam_lr)

loss_history = []

print("\nFase 1: Adam ")

for epoch in range(1, adam_epochs + 1):
    optimizer.zero_grad()

    loss_total, loss_data, loss_phys, loss_u, loss_v, loss_cont, loss_mx, loss_my = pinn_loss(
        X_train_t,
        u_train_t,
        v_train_t
    )

    loss_total.backward()
    optimizer.step()

    loss_history.append([
        epoch,
        loss_total.item(),
        loss_data.item(),
        loss_phys.item(),
        loss_u.item(),
        loss_v.item(),
        loss_cont.item(),
        loss_mx.item(),
        loss_my.item(),
        math.exp(float(log_l1.detach().cpu())),
        math.exp(float(log_l2.detach().cpu()))
    ])

    if epoch % 1000 == 0 or epoch == 1:
        l1 = math.exp(float(log_l1.detach().cpu()))
        l2 = math.exp(float(log_l2.detach().cpu()))
        print(
            f"Epoch {epoch:6d} | "
            f"Loss={loss_total.item():.4e} | "
            f"Data={loss_data.item():.4e} | "
            f"Phys={loss_phys.item():.4e} | "
            f"lambda1={l1:.5f} | lambda2={l2:.5f}"
        )

# Entrenamiento L-BFGS

print("\nFase 2: L-BFGS ")

optimizer_lbfgs = torch.optim.LBFGS(
    params,
    lr=1.0,
    max_iter=lbfgs_max_iter,
    max_eval=lbfgs_max_iter,
    history_size=50,
    tolerance_grad=1e-8,
    tolerance_change=1e-9,
    line_search_fn="strong_wolfe"
)

lbfgs_iter = [0]

def closure():
    optimizer_lbfgs.zero_grad()

    loss_total, loss_data, loss_phys, loss_u, loss_v, loss_cont, loss_mx, loss_my = pinn_loss(
        X_train_t,
        u_train_t,
        v_train_t
    )

    loss_total.backward()

    lbfgs_iter[0] += 1

    if lbfgs_iter[0] % 50 == 0 or lbfgs_iter[0] == 1:
        l1 = math.exp(float(log_l1.detach().cpu()))
        l2 = math.exp(float(log_l2.detach().cpu()))
        print(
            f"L-BFGS {lbfgs_iter[0]:5d} | "
            f"Loss={loss_total.item():.4e} | "
            f"Data={loss_data.item():.4e} | "
            f"Phys={loss_phys.item():.4e} | "
            f"lambda1={l1:.5f} | lambda2={l2:.5f}"
        )

    return loss_total

optimizer_lbfgs.step(closure)

# Parámetros identificados

l1 = math.exp(float(log_l1.detach().cpu()))
l2 = math.exp(float(log_l2.detach().cpu()))
Re = 1.0 / l2

print(" PARÁMETROS IDENTIFICADOS ")
print(f"λ1 = {l1:.6f}   esperado ≈ 1.000000 ")
print(f"  λ2 = {l2:.6f}   esperado ≈ 0.010000 ")
print(f" Re = {Re:.2f}       esperado ≈ 100.00 ")

# VALIDACION DE LOS PUNTOS DE ENTRENAMIENTO

def predict_np(X_np):
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_np, dtype=torch.float32, device=device)
        pred = model(X_t).cpu().numpy()
    return pred


pred_train = predict_np(X_train)

u_pred_train = pred_train[:, 0:1]
v_pred_train = pred_train[:, 1:2]
p_pred_train = pred_train[:, 2:3]

# Errores de velocidad
error_u = np.linalg.norm(u_pred_train - u_train) / np.linalg.norm(u_train)
error_v = np.linalg.norm(v_pred_train - v_train) / np.linalg.norm(v_train)

# Presión: error bruto
error_p_raw = np.linalg.norm(p_pred_train - p_train) / np.linalg.norm(p_train)
MAE_p_raw = np.mean(np.abs(p_pred_train - p_train))

# Presión: error corregido sin offset medio
p_pred_corr = p_pred_train - np.mean(p_pred_train)
p_real_corr = p_train - np.mean(p_train)

error_p_corr = np.linalg.norm(p_pred_corr - p_real_corr) / np.linalg.norm(p_real_corr)
MAE_p_corr = np.mean(np.abs(p_pred_corr - p_real_corr))

print("\nErrores de validación ")
print(f"  L2 u              : {error_u:.4e}")
print(f"  L2 v              : {error_v:.4e}")
print(f"  L2 p bruto        : {error_p_raw:.4e}")
print(f"  MAE p bruto       : {MAE_p_raw:.4e}")
print(f"  L2 p sin offset   : {error_p_corr:.4e}")
print(f"  MAE p sin offset  : {MAE_p_corr:.4e}")

# CAMPOS PREDICHOS EN UN INSTANTE DEL DATASET

t_fixed = 4.0

unique_times = np.unique(data_domain[:, 2])
t_near = unique_times[np.argmin(np.abs(unique_times - t_fixed))]

mask_t = np.isclose(data_domain[:, 2], t_near)

slice_data = data_domain[mask_t]

X_slice = slice_data[:, 0:3]
x_slice = slice_data[:, 0]
y_slice = slice_data[:, 1]

pred_slice = predict_np(X_slice)

u_slice_pred = pred_slice[:, 0]
v_slice_pred = pred_slice[:, 1]
p_slice_pred = pred_slice[:, 2]
p_slice_pred_centered = p_slice_pred - np.mean(p_slice_pred)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

fields = [
    (u_slice_pred, "u predicha"),
    (v_slice_pred, "v predicha"),
    (p_slice_pred_centered, "p inferida centrada")
]

for ax, (field, title) in zip(axes, fields):
    sc = ax.scatter(
        x_slice,
        y_slice,
        c=field,
        s=10,
        cmap="RdBu_r"
    )
    plt.colorbar(sc, ax=ax)
    ax.set_title(f"{title} | t = {t_near:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

plt.suptitle(
    f"PINN HFM — Campos predichos | λ1={l1:.3f}, λ2={l2:.4f}, Re={Re:.1f}",
    fontsize=12
)
plt.tight_layout()
plt.savefig(f"{output_dir}/fig1_hfm_campos_predichos.png", dpi=200, bbox_inches="tight")
plt.show()

# COMPARACION DE LA PRESION OPTENIDA CON LA DEL DATASET

p_real_slice = slice_data[:, 5:6]
p_infer_slice = pred_slice[:, 2:3]

# Comparación sin offset
p_real_plot = p_real_slice - np.mean(p_real_slice)
p_infer_plot = p_infer_slice - np.mean(p_infer_slice)

error_abs = np.abs(p_real_plot - p_infer_plot)

vmin = min(p_real_plot.min(), p_infer_plot.min())
vmax = max(p_real_plot.max(), p_infer_plot.max())

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

sc1 = axes[0].scatter(
    x_slice,
    y_slice,
    c=p_real_plot.flatten(),
    s=10,
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax
)
axes[0].set_title("p real centrada\n(no usada en entrenamiento)")
plt.colorbar(sc1, ax=axes[0])

sc2 = axes[1].scatter(
    x_slice,
    y_slice,
    c=p_infer_plot.flatten(),
    s=10,
    cmap="RdBu_r",
    vmin=vmin,
    vmax=vmax
)
axes[1].set_title("p inferida centrada\n(solo u, v y Navier-Stokes)")
plt.colorbar(sc2, ax=axes[1])

sc3 = axes[2].scatter(
    x_slice,
    y_slice,
    c=error_abs.flatten(),
    s=10,
    cmap="hot_r"
)
axes[2].set_title(f"Error absoluto\nMAE sin offset = {MAE_p_corr:.4e}")
plt.colorbar(sc3, ax=axes[2])

for ax in axes:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

plt.suptitle(
    f"Validación de presión inferida | t = {t_near:.3f} | L2 sin offset = {error_p_corr:.4e}",
    fontsize=12
)
plt.tight_layout()
plt.savefig(f"{output_dir}/fig2_hfm_comparacion_presion.png", dpi=200, bbox_inches="tight")
plt.show()

# EVOLUCION TEMPORAL DE LA PRESION INFERIDA

t_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()

for ax, t_val in zip(axes, t_vals):
    t_sel = unique_times[np.argmin(np.abs(unique_times - t_val))]
    mask_ti = np.isclose(data_domain[:, 2], t_sel)
    data_ti = data_domain[mask_ti]

    X_ti = data_ti[:, 0:3]
    x_ti = data_ti[:, 0]
    y_ti = data_ti[:, 1]

    pred_ti = predict_np(X_ti)
    p_ti = pred_ti[:, 2]
    p_ti_centered = p_ti - np.mean(p_ti)

    sc = ax.scatter(
        x_ti,
        y_ti,
        c=p_ti_centered,
        s=8,
        cmap="RdBu_r"
    )
    plt.colorbar(sc, ax=ax)
    ax.set_title(f"p inferida centrada | t = {t_sel:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")

plt.suptitle("Evolución temporal de la presión inferida por PINN", fontsize=13)
plt.tight_layout()
plt.savefig(f"{output_dir}/fig3_hfm_evolucion_temporal_p.png", dpi=200, bbox_inches="tight")
plt.show()

