import matplotlib.pyplot as plt
import numpy as np

# Dados do gráfico
methods = [
    "18 features (RF)",
    "8 features (RF)",
    "18 features (XGB)",
    "8 features (XGB)",
    "18 features (MLP)",
    "8 features (MLP)",
]

cross_val = [0.6664, 0.6652, 0.6893, 0.6875,0.6870, 0.6804]
test =      [0.6297, 0.6316, 0.6299, 0.6331, 0.6405, 0.6408]

x = np.arange(len(methods))  # posições no eixo X
width = 0.35                 # largura das barras

plt.figure(figsize=(10, 6))

# Barras
plt.bar(x - width/2, cross_val, width, label="CrossValidation", color="#13AFFD")
plt.bar(x + width/2, test,      width, label="Test",            color="#FF8C00")

# Configurações do gráfico
plt.ylabel("Accuracy", fontsize=12)
plt.xlabel("Methods (Models)", fontsize=12)
plt.xticks(x, methods, rotation=35, ha="right")
plt.ylim(0.45, 0.75)

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.legend(loc="upper right")

plt.tight_layout()
plt.show()