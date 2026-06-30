#import mocaloss as mcl
#import polars as pl
mcl = 1
pl = 2

# 1. Crear el modelo indicando el número de iteraciones de Monte Carlo
model = mcl.Model(iterations=10000)

# 2. Configurar las distribuciones para la Variable A según el código
# Si el código es 1 -> Normal; si es 2 -> Lognormal
model.add_unknown_variable(
    name="variable_a",
    conditional_on="codigo",
    mapping={
        1: mcl.dist.Normal(mu=10.0, sigma=2.0),
        2: mcl.dist.Lognormal(mu=2.5, sigma=0.4)
    }
)

# 3. Configurar las distribuciones para la Variable B según el código
# Parámetros diferentes a la Variable A
model.add_unknown_variable(
    name="variable_b",
    conditional_on="codigo",
    mapping={
        1: mcl.dist.Normal(mu=15.0, sigma=3.5),
        2: mcl.dist.Lognormal(mu=3.0, sigma=0.2)
    }
)

# 4. Definir el cálculo del resultado final usando expresiones puras de Polars
model.add_calculations([
    (pl.col("variable_a") + pl.col("variable_b")).alias("resultado")
])

# 5. Cargar el DataFrame base del usuario (que contiene la columna 'codigo') y ejecutar
df = pl.DataFrame({
    "id_edificio": [101, 102, 103],
    "codigo": [1, 2, 1]  # Edificios con diferentes tipologías
})

resultados = model.run(dataset=df)

# 6. Ver el resultado estocástico final
print(resultados)