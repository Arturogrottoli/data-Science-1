# Clase 02: Fundamentos de Python — Guía Completa para el Docente

Esta guía es el **libreto de apoyo para dictar la Clase 02**. Reúne, en un solo lugar y con más profundidad de la que entra en una diapositiva, toda la teoría y todos los ejemplos que aparecen en:

- **`Clase 02.pdf`** — el material teórico oficial de la unidad.
- **`Clase02.html`** — las diapositivas que se proyectan en clase (35 filminas, con el mismo orden que este documento).
- **`clase_2.ipynb`** — el notebook con todos los ejemplos ejecutables, comentados y con teoría ampliada, más un ejemplo resuelto de la Segunda Pre-Entrega al final.

La idea es que si en medio de la clase te falla la memoria sobre un detalle (¿por qué `"5" + 3` da error?, ¿qué iba primero, `and` o `or`?, ¿cómo era la sintaxis de `**kwargs`?), lo encuentres acá explicado con más contexto del que alcanza a mostrar una filmina.

---

## Índice

0. [Mapa rápido de la clase](#0-mapa-rápido-de-la-clase)
1. [Módulo 1 — Variables, Tipos Escalares y Operadores](#módulo-1--variables-tipos-escalares-y-operadores)
2. [Módulo 2 — Colecciones a Fondo](#módulo-2--colecciones-a-fondo)
3. [Módulo 3 — Control de Flujo](#módulo-3--control-de-flujo-decisiones-y-bucles)
4. [Módulo 4 — Funciones Reutilizables](#módulo-4--funciones-reutilizables)
5. [Break del Coder](#break-del-coder)
6. [Módulo 5 — Funciones Avanzadas](#módulo-5--funciones-avanzadas)
7. [Módulo 6 — Ingesta y Radiografía Inicial con Pandas](#módulo-6--ingesta-y-radiografía-inicial-con-pandas)
8. [Módulo 7 — Segunda Pre-Entrega](#módulo-7--segunda-pre-entrega)
9. [Preguntas frecuentes y errores típicos a anticipar](#preguntas-frecuentes-y-errores-típicos-a-anticipar)
10. [Material de la clase](#material-de-la-clase)

---

## 0. Mapa rápido de la clase

| # | Módulo | Slides | Notebook (`clase_2.ipynb`) | Idea central |
|---|--------|--------|------------------------------|---------------|
| 1 | Variables, Tipos Escalares y Operadores | 02–08 | Secciones 2.1–2.3 | Los "átomos" de un programa: dato, variable, tipo, operador |
| 2 | Colecciones a Fondo | 09–12 | Sección 2.2 (listas, tuplas, dicts, sets) | Agrupar muchos escalares con distintas garantías (orden, mutabilidad, unicidad) |
| 3 | Control de Flujo | 13–17 | Sección 2.4 | Que el programa decida y repita solo |
| 4 | Funciones Reutilizables | 18–22 | Sección 2.5 | DRY: escribir la lógica una vez, usarla mil veces |
| — | **Break del Coder** | 18 (movida antes del Módulo 4) | — | Corte de ~10 minutos |
| 5 | Funciones Avanzadas | 23–26 | Sección 2.5 (continuación) | `*args`, `**kwargs`, scope, lambda |
| 6 | Ingesta y Radiografía con Pandas | 27–31 | Sección 2.6 | Primer contacto con DataFrames reales |
| 7 | Segunda Pre-Entrega | 32–33 | Sección 2.9 (ejemplo resuelto) | Consigna + ejemplo completo de referencia |

> **Nota sobre el break**: en las diapositivas, el "Break del Coder" está ubicado **justo antes** del divisor de "Funciones Reutilizables" (después de `break`/`continue`, antes de arrancar Funciones). Es un buen corte natural porque cierra Python "básico" (tipos, colecciones, control de flujo) y abre la segunda mitad (funciones + Pandas).

---

## Módulo 1 — Variables, Tipos Escalares y Operadores

### 1.1 Variable vs. Dato

**Concepto**: antes de poder operar con información, necesitamos un lugar donde guardarla. Un **dato** es la unidad de información en sí (`25`, `"Juan"`, `True`); una **variable** es el nombre que le ponemos para poder referirnos a ese dato más adelante.

**Analogía para usar en clase**: la variable es como una etiqueta adhesiva y el dato es el objeto al que se la pegás. Si pegás la etiqueta "Precio" a un billete de 10 dólares, la etiqueta te permite encontrar ese valor después, pero la etiqueta no *es* el valor.

**Tipado dinámico**: a diferencia de lenguajes como Java o C#, en Python **no declarás el tipo** de la variable. El intérprete mira el dato en el momento de la asignación y decide automáticamente qué tipo es.

```python
edad = 30
print(f"edad = {edad} -> {type(edad)}")   # edad = 30 -> <class 'int'>

edad = "treinta"  # la misma variable ahora apunta a un string
print(f"edad = {edad} -> {type(edad)}")   # edad = treinta -> <class 'str'>
```

**Punto fino para mencionar en clase (pregunta frecuente)**: que Python sea *dinámico* no significa que sea *débil*. Python es **fuertemente tipado**: no mezcla tipos incompatibles sin una conversión explícita. `"5" + 3` lanza `TypeError`, a diferencia de JavaScript, que lo concatenaría silenciosamente como `"53"`.

**Asignación múltiple** (útil para mostrar que Python tiene azúcar sintáctica que otros lenguajes no tienen):

```python
x, y, z = 1, 2, 3          # una variable por valor
a = b = c = 0               # las tres apuntan al mismo valor inicial
a, b = 10, 20
a, b = b, a                  # swap sin variable auxiliar -> a=20, b=10
```

---

### 1.2 Tipos Escalares: los átomos de la información

**Concepto**: un tipo escalar es un valor atómico, indivisible — la unidad mínima de información. "Escalar" **no significa pequeño**, significa *un solo valor a la vez* (esto se retoma en 1.4).

| Tipo | Qué representa | Ejemplo en Data Science |
|---|---|---|
| `int` | Números sin decimales | Nº de transacciones, edad en años cumplidos |
| `float` | Números con decimales | Precios, temperatura, probabilidades |
| `str` | Cadenas de texto | Nombres de categorías, emails, reseñas |
| `bool` | Solo `True` o `False` | ¿Es cliente premium?, ¿la columna tiene nulos? |

```python
entero = 42
decimal = 3.14159
texto = "Hola mundo"
booleano = True
nada = None
```

**Detalle técnico para mencionar si preguntan por qué `0.1 + 0.2 != 0.3`**: los `int` en Python tienen precisión arbitraria (no se desbordan como en C o Java, podés tener un entero de 200 dígitos sin problema). Los `float`, en cambio, usan el estándar de punto flotante IEEE 754 y pueden acumular pequeños errores de redondeo. Para cálculos financieros exactos existe el módulo `decimal` (mencionarlo, no hace falta profundizar).

---

### 1.3 Cadenas como Secuencia: el "Collar de Perlas"

**Concepto**: aunque un `str` se percibe como un dato simple (el nombre de una persona), Python lo trata internamente como una **secuencia** de caracteres. Podés medir el string completo o pedir un carácter puntual, igual que contás perlas en un collar o sacás una en particular.

```python
ciudad = "Madrid"
print(len(ciudad))       # 6 -> cuántas "perlas" tiene
print(ciudad[0])         # 'M' -> primera letra
print(ciudad[-1])        # 'd' -> última letra
print(ciudad[0:3])       # 'Mad' -> slicing (sub-cadena)
```

**Por qué importa en Data Science**: esta naturaleza de secuencia es la que permite limpiar nombres (quitar espacios), extraer códigos de área de números telefónicos o buscar palabras clave en un tweet — todo el trabajo de limpieza de texto se apoya en tratar al string como algo indexable y medible.

**Inmutabilidad (para mencionar)**: los strings son secuencias **inmutables** — no se puede hacer `ciudad[0] = "m"` (lanza `TypeError`). Métodos como `.upper()` o `.strip()` no modifican el string original, devuelven uno **nuevo**.

---

### 1.4 Escalares vs. Colecciones

**Error común de los alumnos que conviene anticipar**: confundir un dato "grande" con una colección. En programación, escalar **no significa pequeño**, significa *un solo valor en un momento dado*.

- **Escalar**: `temperatura = 24.5` → un solo valor.
- **Colección**: `temperaturas_semana = [24.5, 22.0, 26.1, 23.4]` → una estructura que agrupa múltiples valores.

**Analogía**: un escalar es como una sola hoja de papel con un dato escrito; una colección es como una carpeta que agrupa muchas hojas relacionadas.

**Por qué importa (para funciones y scope, más adelante)**: al pasar un escalar a una función, se pasa "por valor" (una copia). Al pasar una colección mutable, se pasa una referencia al mismo objeto en memoria — modificarla dentro de la función puede afectar al original. Este es el gancho perfecto para retomar el tema cuando lleguen al Módulo 5 (scope global).

---

### 1.5 Operadores Aritméticos y de Comparación

**Aritméticos**: `+`, `-`, `*`, `/` (división decimal), `//` (división entera), `%` (módulo/resto), `**` (potencia).

```python
7 / 2    # 3.5   -> división decimal, siempre devuelve float
7 // 2   # 3     -> división entera, descarta el resto
7 % 2    # 1     -> resto/módulo (muy usado para saber si un número es par: n % 2 == 0)
2 ** 3   # 8     -> potencia
```

**Comparación**: `==`, `!=`, `>`, `<`, `>=`, `<=` — **siempre devuelven un booleano**.

```python
5 == 5   # True
5 != 3   # True
```

> **El error de sintaxis más común en principiantes**: confundir `=` (asignar) con `==` (comparar). `if edad = 18` es un error de sintaxis en Python; hay que escribir `if edad == 18`. Vale la pena remarcarlo fuerte en clase porque va a aparecer una y otra vez en los ejercicios.

**Un mismo operador, distintos comportamientos** (dato interesante para dar contexto): el símbolo `+` no siempre suma números — en strings concatena (`"a" + "b"` → `"ab"`) y en listas une (`[1, 2] + [3]` → `[1, 2, 3]`). Esto se llama *sobrecarga de operadores* y es la razón por la que más adelante, en Pandas y NumPy, se puede escribir `columna_a + columna_b` sin bucles.

---

### 1.6 Operadores Lógicos y Conversión de Tipos

**Lógicos**: `and`, `or`, `not` — permiten combinar varias comparaciones.

- **`and` (estricto)**: `True` solo si **todas** las condiciones son ciertas. Caso de uso: *¿el usuario inició sesión Y supera su límite habitual?* → alerta de fraude.
- **`or` (flexible)**: `True` si **al menos una** se cumple. Caso de uso: *¿el email está vacío O no tiene arroba?* → registro inválido.
- **`not`**: invierte el booleano. Caso de uso: *si el registro NO es nulo, procesarlo*.

**Evaluación de cortocircuito (short-circuit, para quien pregunte "por qué no explota")**: Python evalúa `and`/`or` de izquierda a derecha y se detiene apenas puede determinar el resultado. En `a and b`, si `a` es `False`, ni siquiera evalúa `b`. Por eso `df is not None and len(df) > 0` nunca intenta medir la longitud de un `None`: el `and` corta antes de llegar ahí.

**Conversión de tipos (casting)**: la herramienta para "limpiar" datos que llegan en el formato incorrecto — típicamente un sensor o un CSV que trae números como texto.

```python
int("10")      # 10
float("10.5")  # 10.5
str(100)       # "100"
bool(1)        # True
```

---

### 1.7 Precedencia de Operadores y Errores Comunes

Python resuelve expresiones largas siguiendo un orden fijo (igual que la matemática de la escuela):

1. Paréntesis `()`
2. Potencias `**`
3. Multiplicación, División, Módulo
4. Suma y Resta
5. Comparaciones (`==`, `>`, etc.)
6. Operadores lógicos (`not`, `and`, `or`)

**Ejemplo clásico para el pizarrón**: `10 + 5 * 2` da `20`, no `30`, porque `*` se resuelve antes que `+`. Si el resultado buscado era `30`, hay que forzar el orden con paréntesis: `(10 + 5) * 2`.

**Tres errores para anticipar activamente en clase:**
1. **Comparar tipos incompatibles**: `"5" > 3` falla — hay que convertir un lado antes de comparar.
2. **`=` vs. `==`**: ya mencionado en 1.5, pero conviene repetirlo acá también.
3. **Strings sin comillas**: escribir `nombre = Juan` hace que Python busque una variable llamada `Juan` (y probablemente falle con `NameError`) en lugar de asignar el texto `"Juan"`.

**Dato curioso para dar más contexto**: Python permite **comparaciones encadenadas**: `0 <= edad < 18` es válido y equivale a `0 <= edad and edad < 18`, pero es más legible. Es una particularidad de Python que no existe en la mayoría de los otros lenguajes — vale la pena mostrarla como ejemplo de la "elegancia" del lenguaje.

---

## Módulo 2 — Colecciones a Fondo

### 2.1 Listas vs. Tuplas: el dilema de la mutabilidad

La diferencia técnica más importante en esta sección es si una colección puede cambiar después de creada.

- **Listas `[]` (mutables)**: pueden crecer, achicarse y modificar sus elementos. Ideales para flujos de datos dinámicos (ej. una lista de ventas del día que se va actualizando).
- **Tuplas `()` (inmutables)**: una vez creadas, no se pueden modificar. En Data Science se usan para **proteger la integridad del dato** — si una función procesa una fila de una base de datos como `(id_usuario, fecha, dni)`, estructurarla como tupla garantiza que ningún bug altere el DNI por accidente.

```python
# Listas: dinámicas
ventas_del_dia = [1500, 2300, 4200]
ventas_del_dia.append(3100)  # permitido

# Tuplas: protegidas
registro_cliente = (1024, "2026-05-29", "Juan Pérez")
# registro_cliente[1] = "2026-06-01"  # TypeError: 'tuple' object does not support item assignment
```

**Para remarcar en clase**: la elección entre lista y tupla no es solo de sintaxis, es una **decisión de diseño**: "¿quiero que esto se pueda modificar más adelante o no?"

---

### 2.2 Diccionarios Anidados: el modelo mental de un JSON

**Concepto**: cuando se consumen datos de una API web (Spotify, clima, redes sociales), la información no llega en una tabla prolija — llega como **JSON**, que en Python se traduce directamente en un **diccionario anidado** (diccionarios dentro de diccionarios, o listas dentro de diccionarios).

**La técnica clave**: avanzar "capa por capa" usando las llaves (`keys`) o índices.

```python
api_response = {
    "status": "success",
    "data": {
        "cliente_id": 5582,
        "historial_compras": [
            {"producto": "Teclado Mecánico", "precio": 85},
            {"producto": "Mouse Pad XL", "precio": 30}
        ],
        "localizacion": {"pais": "Argentina", "provincia": "Buenos Aires"}
    }
}

primer_precio = api_response["data"]["historial_compras"][0]["precio"]
print(primer_precio)  # 85
```

**Advertencia para dar en clase**: acceder con `[...]` a una clave que no existe lanza `KeyError` y corta la ejecución. Para datos reales de una API (donde los campos pueden faltar) conviene usar `.get("clave", valor_por_defecto)` en cada nivel en lugar de asumir que la clave siempre está.

---

### 2.3 Sets: el arma secreta para la limpieza de datos

**Concepto**: un `set` es una colección **desordenada de elementos únicos**. En Data Science se usa para dos tareas muy puntuales y muy frecuentes:

1. **Deduplicación instantánea**: eliminar registros duplicados en una sola línea.
2. **Operaciones de conjuntos**: cruzar listas de usuarios (¿quiénes compraron en la campaña A *y* en la B?).

```python
# 1. Eliminar duplicados
emails = ["juan@gmail.com", "ana@gmail.com", "juan@gmail.com", "pedro@gmail.com"]
emails_unicos = list(set(emails))

# 2. Intersección de campañas
clientes_cybermonday = {"user_1", "user_2", "user_3"}
clientes_blackfriday = {"user_3", "user_4", "user_5"}
compraron_en_ambos = clientes_cybermonday.intersection(clientes_blackfriday)  # {'user_3'}
```

**Por qué son tan rápidos (dato técnico interesante)**: internamente un `set` usa una tabla hash, igual que las claves de un diccionario. Preguntar `x in mi_set` es prácticamente instantáneo, mucho más rápido que buscar en una lista grande. La contrapartida: los elementos de un set deben ser *hasheables* — no se puede meter una lista dentro de un set, pero sí una tupla.

---

## Módulo 3 — Control de Flujo: Decisiones y Bucles

**Contexto para abrir el módulo**: en el mundo real, los datos perfectos no existen. Un pipeline recibe miles de transacciones diarias: algunas con precios negativos por un bug, otras con campos vacíos, otras de clientes VIP que necesitan un tratamiento especial. El control de flujo es lo que permite programar scripts autónomos que toman esas decisiones, filtran la "basura" y transforman datos en información útil, sin intervención manual.

### 3.1 if, elif, else

Un condicional es un **bifurcador de caminos lógicos**.

- **`if`**: evalúa la condición principal.
- **`elif`**: plan alternativo si la anterior fue falsa; se pueden encadenar tantos como haga falta.
- **`else`**: caso por defecto — captura todo lo que no cumplió ninguna condición anterior (útil para atrapar errores imprevistos o registros sospechosos).

```python
score_crediticio = 650

if score_crediticio >= 750:
    estado = "Aprobado Automático"
elif score_crediticio >= 600:
    estado = "Revisión Manual"
else:
    estado = "Rechazado"

print(estado)  # Revisión Manual
```

**Dato para dar contexto**: Python no tuvo históricamente una instrucción `switch/case` — una cadena larga de `elif` cumple ese rol. Desde Python 3.10 existe `match/case` como alternativa más moderna, pero `if/elif/else` sigue siendo el estándar en la inmensa mayoría del código de ciencia de datos, así que es lo que enseñamos.

### 3.2 Álgebra Booleana: and, or, not aplicados

Reforzar con casos de uso concretos de negocio (ya vistos en 1.6, pero ahora aplicados a filtros de datasets):

- **`and`**: filtro estricto — todas las condiciones deben cumplirse.
- **`or`**: filtro flexible — alcanza con que se cumpla una.
- **`not`**: invierte una condición (típicamente para "no nulo", "no vacío").

### 3.3 Bucles for y while

- **`for`**: itera sobre colecciones (listas, diccionarios, rangos). Es *predictivo* — sabemos exactamente cuándo empieza y cuándo termina, porque recorre el largo de la colección. Es "el rey" de Data Science porque la inmensa mayoría de las tareas de transformación son recorrer una colección elemento por elemento.
- **`while`**: se ejecuta **mientras** una condición sea verdadera. En Data Science se usa menos para recorrer datos estructurados y más para interactuar con el entorno exterior (reintentar una conexión a una API, algoritmos iterativos que deben repetirse hasta que el error numérico sea chico).

```python
precios_usd = [10, 45, 120, 5]
tipo_cambio = 1000
precios_ars = []

for precio in precios_usd:
    precios_ars.append(precio * tipo_cambio)

print(precios_ars)  # [10000, 45000, 120000, 5000]
```

**Riesgo del `while` para remarcar**: si la condición nunca se vuelve falsa, el bucle corre para siempre (bucle infinito) y cuelga el notebook. Siempre hay que asegurarse de que algo dentro del bucle modifique la variable que controla la condición (`contador += 1`).

### 3.4 break y continue

- **`break`**: salida de emergencia — interrumpe el bucle de inmediato, sin importar cuántos elementos queden. Caso de uso: buscás si existe *al menos un* registro corrupto en un archivo masivo; en cuanto lo encontrás, cortás — no tiene sentido seguir procesando el resto.
- **`continue`**: detiene la iteración actual y salta a la siguiente, sin romper el script. Caso de uso típico de *data cleaning*: si una fila tiene datos nulos, la salteás con `continue` y seguís con el resto sin abortar todo el proceso.

```python
registro_ventas = [150, -20, 300, 0, 450]  # el -20 es un error sistémico

for venta in registro_ventas:
    if venta <= 0:
        continue  # ignora negativos/cero
    impuesto = venta * 0.21
    print(f"Venta: {venta} | Impuesto: {impuesto}")
```

**Alcance limitado (para mencionar si preguntan por bucles anidados)**: `break` y `continue` solo afectan al bucle **más interno** en el que están escritos, no cortan automáticamente los bucles externos. Si hace falta salir de varios niveles a la vez, una técnica común es envolver la lógica en una función y usar `return`.

---

## Módulo 4 — Funciones Reutilizables

**Contexto para abrir el módulo**: si tuviéramos que limpiar los datos de 50 sucursales copiando y pegando el mismo código 50 veces, terminaríamos con un programa gigante, difícil de leer y propenso a errores. Las funciones son la herramienta para aplicar el principio **DRY** (*Don't Repeat Yourself*).

### 4.1 Qué es una función y por qué reutilizarla

Una función es un bloque de código autónomo: le das materia prima (parámetros), hace un proceso, te entrega un resultado (`return`).

**Cuatro razones para insistir en clase sobre por qué son vitales en Data Science:**
1. **Modularidad**: dividen un problema complejo (un pipeline completo de limpieza) en piezas manejables.
2. **Reutilización**: se escribe la lógica una vez y se usa mil veces, incluso en proyectos futuros.
3. **Mantenibilidad**: si hay un error en la lógica de transformación, se corrige en un solo lugar y el cambio se aplica en todo el programa.
4. **Abstracción**: quien *usa* la función no necesita saber *cómo* hace el cálculo, solo *qué* necesita y *qué* le entrega (como manejar un auto sin saber de mecánica).

**Nota de buenas prácticas para reforzar**: idealmente una función es *pura* — depende solo de sus parámetros y no modifica nada fuera de sí misma. Son más fáciles de testear y depurar, algo muy valioso en pipelines de limpieza.

### 4.2 def, parámetros, argumentos y return

```python
def normalizar_categoria(texto):
    return texto.strip().lower()

print(normalizar_categoria(" ELECTRONICA "))  # "electronica"
```

- **`texto`** es el **parámetro** (el molde); **`" ELECTRONICA "`** es el **argumento** (el valor real que se pasa).
- **Argumentos posicionales vs. con nombre**: `crear_etiqueta("Camisa", 25)` asigna por orden; `crear_etiqueta(precio=40, producto="Pantalón")` asigna explícitamente por nombre y por eso el orden deja de importar. Usar argumentos con nombre es más seguro cuando una función tiene muchos parámetros de configuración.

**El punto que más confunde a los principiantes — `print()` vs. `return`:**

| | `print()` | `return` |
|---|---|---|
| Qué hace | Muestra el valor en pantalla | Entrega el valor al programa |
| ¿Se puede reutilizar el resultado? | No — el programa lo "olvida" | Sí — se puede guardar en una variable |
| Analogía | El repartidor te muestra la pizza y se la lleva | El repartidor te entrega la caja: ahora es tuya |

```python
def sumar_sin_return(a, b):
    print(a + b)          # solo lo muestra

def sumar_con_return(a, b):
    return a + b           # lo entrega para reutilizarlo

resultado_1 = sumar_sin_return(2, 3)   # imprime 5, pero...
resultado_2 = sumar_con_return(2, 3)

print(resultado_1)  # None -> ¡no se pudo reutilizar!
print(resultado_2)  # 5
```

Si una función no tiene `return` explícito, Python devuelve `None` por defecto — este es el motivo por el que `resultado_1` es `None` arriba.

### 4.3 Valores por defecto, múltiples retornos y scope

**Valores por defecto**: cuando un parámetro casi siempre tiene el mismo valor, se puede predefinir para no escribirlo siempre.

```python
def convertir_moneda(monto, tasa=0.92):
    return monto * tasa

convertir_moneda(100)        # 92.0  -> usa la tasa por defecto
convertir_moneda(100, 0.85)  # 85.0  -> la sobrescribe
```

**Múltiples retornos**: en análisis de datos, a veces se necesita que una función entregue varios resultados a la vez. Python permite devolver varios valores separados por coma — internamente se empaquetan como una tupla.

```python
def obtener_extremos(lista_precios):
    return min(lista_precios), max(lista_precios)

p_min, p_max = obtener_extremos([10, 50, 2, 100])
```

**Scope (ámbito)**: las variables creadas dentro de una función son **locales** — solo existen mientras la función se ejecuta y desaparecen al terminar. Esto evita "ensuciar" el programa principal con nombres de variables temporales.

```python
def mi_funcion():
    variable_local = "Soy secreta"
    print(variable_local)

mi_funcion()
# print(variable_local)  # NameError: no existe fuera de la función
```

---

## Break del Coder

Corte de ~10 minutos, ubicado justo después de `break`/`continue` y **antes** de arrancar el divisor de "Funciones Reutilizables". Es un buen momento porque cierra "Python básico" y da pie a la segunda mitad de la clase.

---

## Módulo 5 — Funciones Avanzadas

### 5.1 *args y **kwargs

Las funciones "maduras" no solo reciben variables fijas: pueden aceptar una cantidad **indefinida** de argumentos.

- **`*args`**: captura cualquier cantidad de argumentos **posicionales** adicionales.
- **`**kwargs`**: captura cualquier cantidad de argumentos **con nombre** adicionales, sin declararlos de antemano.

```python
def formatear_metricas(valor, moneda="USD", **detalles_extra):
    texto_base = f"Monto: {valor} {moneda}"
    if detalles_extra.get("notificar_gerencia") == True:
        texto_base += " ⚠️ [ALERTA ALTA GERENCIA]"
    return texto_base

print(formatear_metricas(5000))
# Monto: 5000 USD

print(formatear_metricas(750000, moneda="ARS", notificar_gerencia=True))
# Monto: 750000 ARS ⚠️ [ALERTA ALTA GERENCIA]
```

**Por qué importa**: comprender `*args`/`**kwargs` es la base para entender cómo están programadas muchas librerías de Machine Learning (scikit-learn, por ejemplo, usa `**kwargs` constantemente para pasar hiperparámetros).

**Aclaración útil si preguntan**: `args` y `kwargs` son solo **nombres convencionales** — lo que realmente le indica a Python que empaquete los argumentos son los símbolos `*` y `**`. Técnicamente se podría escribir `*valores` o `**opciones`, pero seguir la convención hace el código más reconocible para otros programadores.

### 5.2 El peligro del scope global

**Regla de oro para remarcar con énfasis**: nunca modifiques una variable global directamente dentro de una función de limpieza. Si lo hacés, perdés la trazabilidad de cómo cambiaron tus datos (efectos secundarios difíciles de rastrear).

```python
dataset_global = [10, 20, 30]

def duplicar_datos(lista_datos):   # lista_datos es un parámetro local
    copia_limpia = [x * 2 for x in lista_datos]
    return copia_limpia

nuevos_datos = duplicar_datos(dataset_global)
print(dataset_global)  # [10, 20, 30] -> el original está a salvo
print(nuevos_datos)    # [20, 40, 60] -> el resultado esperado
```

**Nota técnica**: Python permite modificar una variable global desde dentro de una función usando la palabra clave `global nombre_variable`, pero esto vuelve el código impredecible en proyectos grandes. La práctica recomendada (y la que se enseña en esta clase) es siempre **pasar y retornar** los datos explícitamente, como en el ejemplo de arriba.

### 5.3 Funciones lambda

Una función **lambda** es una función anónima que se escribe en una sola línea. No reemplaza a `def` — se usa para tareas ultra-específicas de "un solo uso", cuando armar una función completa sería exagerado.

```python
calcular_iva = lambda precio: precio * 1.21
print(calcular_iva(100))  # 121.0

precios_viejos = [100, 250, 400]
precios_actualizados = list(map(lambda x: x * 1.10, precios_viejos))
print(precios_actualizados)  # [110.0, 275.0, 440.0]
```

**Combinación clave con `map()` y `filter()`**: `map()` aplica una transformación a toda una lista; `filter()` se queda solo con los elementos que cumplen una condición. Esta lógica es **la antesala directa del método `.apply()` de Pandas**, que se ve en el módulo siguiente — vale la pena decirlo explícitamente en clase para que hagan la conexión.

**Limitación importante para aclarar**: una lambda solo puede contener **una expresión** — no admite bloques `if` con varias líneas, bucles, ni múltiples sentencias. Si la lógica necesita varios pasos o un `if/elif/else` completo, es momento de usar una función `def` normal.

---

## Módulo 6 — Ingesta y Radiografía Inicial con Pandas

**Contexto para abrir el módulo**: hasta acá se manipularon datos con listas y diccionarios nativos. Cuando se trabaja con miles o millones de filas, hace falta una herramienta profesional: **Pandas**, la librería de código abierto más usada del mundo para análisis y manipulación de datos.

### 6.1 ¿Qué es un DataFrame? Ingesta de datos

Un **DataFrame** es una tabla bidimensional (filas y columnas) con índices — muy similar a una hoja de cálculo de Excel o a una tabla SQL, pero optimizada para operaciones matemáticas y lógicas a alta velocidad.

```python
import pandas as pd

df = pd.read_csv("mi_dataset.csv")
# df = pd.read_csv("mi_dataset.csv", sep=";")  # si el separador no es coma

df = pd.read_excel("mi_dataset.xlsx", sheet_name="Ventas2026")
```

**Dato técnico para dar más contexto**: cada columna de un DataFrame es en realidad un objeto `pandas.Series` — una lista con índice, parecida a un diccionario ordenado. Un DataFrame es, literalmente, una colección de Series que comparten el mismo índice de filas.

### 6.2 Primer ojo a la tabla: head, tail y display

- **`df.head(n)`**: primeras `n` filas (5 por defecto). Ideal para verificar que las columnas se cargaron bien.
- **`df.tail(n)`**: últimas `n` filas. Útil para chequear si el archivo se cortó mal o tiene filas vacías al final.
- **`display(df.head())`** (en Jupyter/Colab) renderiza una tabla interactiva y estilizada, mucho más legible que `print(df.head())`, que muestra texto plano.

**Tip adicional para dar en clase**: `df.sample(n)` devuelve `n` filas aleatorias — útil para chequear que los datos "del medio" también se vean razonables, porque `head`/`tail` solo muestran los extremos, que a veces están ordenados de una forma particular y no representan bien al resto del dataset.

### 6.3 La radiografía del dataset: shape, info(), describe()

- **`df.shape`**: es un **atributo** (sin paréntesis), devuelve una tupla `(filas, columnas)`.
- **`df.info()`**: probablemente el comando más importante del diagnóstico inicial — nombre de cada columna, cantidad de valores no nulos y tipo de dato.
- **`df.describe()`**: resumen estadístico de las columnas numéricas (`count`, `mean`, `std`, `min`, percentiles 25/50/75%, `max`) — la "foto panorámica" de cómo se distribuyen los datos.

**Ejemplo de lectura de un `describe()` para trabajar en el pizarrón** (el mismo que usa la diapositiva 30):

| Métrica | Edad | Monto_Compra_USD |
|---|---|---|
| count | 5000.00 | 4850.00 |
| mean | 34.20 | 120.50 |
| std | 10.50 | 340.20 |
| min | -5.00 | 10.00 |
| max | 118.00 | 15000.00 |

Insights para guiar la lectura en voz alta:
- **`count`** distinto entre columnas (`5000` vs. `4850`) → hay 150 filas con `Monto_Compra_USD` vacío.
- **`min` de Edad en `-5`** → físicamente imposible, error de carga a corregir.
- **`max` de Edad en `118`** → sospechoso, posible dato mal cargado.
- **`std` alta en Monto_Compra_USD (340)** → los montos de gasto son muy dispersos: hay clientes que gastan poco y clientes que gastan muchísimo.

**Nota extra**: `df.describe()` por defecto solo resume columnas **numéricas**. Para ver un resumen de columnas de texto/categóricas (valores únicos, más frecuente, etc.) se usa `df.describe(include="object")`.

### 6.4 Datos faltantes: isnull().sum() y % de nulos

```python
print(df.isnull().sum())  # cuenta los NaN por columna

# El truco "pro": True vale 1 y False vale 0, así que el promedio da directamente el %
porcentaje_nulos = df.isnull().mean() * 100

# Tabla de diagnóstico, ordenada de mayor a menor
reporte_nulos = porcentaje_nulos.sort_values(ascending=False)
df_nulos = pd.DataFrame(reporte_nulos, columns=["Porcentaje de Nulos (%)"])
display(df_nulos)
```

**Distinción importante para remarcar**: `isnull()` solo **diagnostica** dónde están los huecos. La estrategia para tratarlos (`dropna()` para eliminar filas, `fillna()` para rellenar con media/mediana/moda, o dejar el nulo si es informativo) depende del contexto de negocio y se define en la etapa de limpieza — no en el diagnóstico. Es un buen punto para cerrar el módulo dejando claro que "diagnosticar" y "decidir qué hacer" son dos pasos distintos.

---

## Módulo 7 — Segunda Pre-Entrega

### Qué deben entregar los alumnos

La actividad pide un notebook (`.ipynb`) estructurado en **4 bloques obligatorios**, aplicados sobre el dataset que cada alumno eligió en la Pre-entrega 1:

| Bloque | Contenido |
|---|---|
| **1. Ingesta y Primer Vistazo** | Importar Pandas, cargar el dataset en `df`, mostrar las primeras 5 filas estilizadas con `display()`. |
| **2. Radiografía Técnica** | `df.shape`, `df.info()`, `df.describe()` + una celda de Markdown con **mínimo 2 hallazgos o anomalías** (valores imposibles, desviaciones muy altas, outliers). |
| **3. Diagnóstico de Datos Faltantes** | Ranking de columnas por porcentaje de nulos, de mayor a menor. |
| **4. Lógica Algorítmica y Funciones Personalizadas** | Una función `def` con `if/elif/else` basada en una regla de negocio del proyecto del alumno, **más** una función `lambda`, ambas **aplicadas al dataset** (idealmente con `.apply()`). |

### Ejemplo resuelto de referencia

En `clase_2.ipynb`, la sección **"2.9 Ejemplo Completo y Explicado: Segunda Pre-Entrega (Resuelto)"** contiene un ejemplo end-to-end sobre un dataset sintético de ventas de e-commerce (con errores intencionales, como un dataset real), resolviendo los 4 bloques de punta a punta:

- Ingesta con `pd.DataFrame` (comentando cómo sería con `pd.read_csv`).
- Radiografía técnica + celda de hallazgos (edad negativa, outlier en monto, tipos de dato).
- % de nulos ordenado.
- Una función `clasificar_gasto()` con `if/elif/else` **y** una `lambda` de limpieza de texto, ambas aplicadas al DataFrame con `.apply()`.

**Sugerencia para dar en clase**: proyectar esa sección del notebook como el "modelo de respuesta" y remarcar explícitamente los 3 puntos que el ejemplo deja como tarea para adaptar: reemplazar el dataset, adaptar la regla de negocio de la función, y adaptar la transformación de la lambda a las columnas propias de cada alumno.

---

## Preguntas frecuentes y errores típicos a anticipar

- **"¿Por qué `"5" > 3` no funciona?"** → Python es fuertemente tipado, no compara texto con número sin conversión explícita (ver 1.6/1.7).
- **"Puse `if edad = 18` y no anda."** → confundieron asignación (`=`) con comparación (`==`) (ver 1.5).
- **"Mi función devuelve `None`."** → seguramente usaron `print()` en vez de `return` dentro de la función (ver 4.2).
- **"Modifiqué la lista adentro de la función y se cambió afuera también."** → gancho perfecto para explicar mutabilidad y paso por referencia (ver 1.4 y 5.2).
- **"¿Cuál uso, lista o tupla?"** → depende de si el dato necesita protegerse de modificaciones accidentales (ver 2.1).
- **"¿`for` o `while`?"** → `for` si sabés de antemano cuántas veces vas a iterar (colecciones); `while` si dependés de una condición externa (ver 3.3).
- **"Mi lambda no anda con un `if` completo."** → las lambdas solo admiten una expresión; para lógica de varios pasos hace falta `def` (ver 5.3).
- **"`df.describe()` no me muestra mis columnas de texto."** → por defecto solo resume numéricas; hace falta `include="object"` (ver 6.3).

---

## Material de la clase

| Archivo | Qué es |
|---|---|
| `Clase 02.pdf` | Material teórico oficial de la unidad (fuente original de esta guía). |
| `Clase02.html` | Diapositivas para proyectar en clase (35 filminas). Abrir en el navegador; navegación con flechas del teclado o los botones inferiores. |
| `clase_2.ipynb` | Notebook con teoría ampliada + todos los ejemplos ejecutables + el ejemplo resuelto de la Segunda Pre-Entrega (sección 2.9). Es el material que se comparte con los alumnos. |
| `material/` | Carpeta con recursos adicionales (PPT, ejercicios propuestos, docx original). |

**Cómo usar esta guía durante la clase**: cada sección de este documento sigue el mismo orden que las diapositivas y el notebook, así que podés ir alternando entre proyectar la filmina/notebook correspondiente y volver acá si necesitás recordar un matiz, un dato de contexto o una pregunta frecuente para anticipar.
