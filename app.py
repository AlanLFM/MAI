from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if not os.path.exists("static"):
        os.makedirs("static")

    formula_pc1 = ""
    formula_pc2 = ""

    if request.method == "POST":
        filtro = request.form.get("filtro", "TODOS")  # Obtener filtro del formulario

        df = pd.read_csv("./star_classification.csv")
        df = df[['u', 'g', 'r', 'i', 'z', 'class']].head(10000)

        if filtro != "TODOS":
            df = df[df["class"] == filtro]  # Filtrar por tipo seleccionado

        X = df[['u', 'g', 'r', 'i', 'z']]
        X_scaled = StandardScaler().fit_transform(X)

        pca = PCA(n_components=5)
        X_pca = pca.fit_transform(X_scaled)

        pc1 = pca.components_[0]
        pc2 = pca.components_[1]

        formula_pc1 = f"PC1 = {pc1[0]:.2f}·u + {pc1[1]:.2f}·g + {pc1[2]:.2f}·r + {pc1[3]:.2f}·i + {pc1[4]:.2f}·z"
        formula_pc2 = f"PC2 = {pc2[0]:.2f}·u + {pc2[1]:.2f}·g + {pc2[2]:.2f}·r + {pc2[3]:.2f}·i + {pc2[4]:.2f}·z"

        df_pca = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
        df_pca['class'] = df['class']

        # Colores personalizados
        color_map = {'STAR': 'yellow', 'GALAXY': 'purple', 'QSO': 'blue'}

        plt.style.use('dark_background')
        plt.figure(figsize=(10, 6))

        if filtro == "TODOS":
            for clase, color in color_map.items():
                subset = df_pca[df_pca['class'] == clase]
                plt.scatter(subset['PC1'], subset['PC2'], c=color, label=clase,
                            s=60, alpha=0.6, edgecolors='white')
        else:
            color = color_map.get(filtro, 'white')
            plt.scatter(df_pca['PC1'], df_pca['PC2'], c=color, label=filtro,
                        s=60, alpha=0.6, edgecolors='white')

        plt.title('Distribución PCA de objetos astronómicos')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig("static/pca_plot.png")
        plt.close()

        # Gráfico de varianza explicada
        plt.figure(figsize=(8, 4))
        plt.style.use('default')
        plt.bar(x=[f'PC{i+1}' for i in range(5)],
                height=pca.explained_variance_ratio_ * 100,
                color='skyblue')
        plt.ylabel('% de varianza explicada')
        plt.title('Importancia de cada componente principal')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig("static/eigen_plot.png")
        plt.close()

    return render_template("index.html", formula_pc1=formula_pc1, formula_pc2=formula_pc2)

if __name__ == "__main__":
    app.run(debug=True)
