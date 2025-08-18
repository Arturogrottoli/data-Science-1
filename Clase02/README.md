# 🐍 Clase 02: Fundamentos de Python para Ciencia de Datos

### **1. Introducción a Python**
[GUIA DE CLASE](https://docs.google.com/presentation/d/1jMeDrYaVZE7IYGxo6fF4dGYgT-KHFZJmzr-vAKvVGes/edit#slide=id.g2204f2a9531_0_0)

### 2.1 Python: Concepto, Variables y Asignación
### 2.2 Tipos de Datos en Python  
### 2.3 Operadores, Objetos y Punteros en Python
### 2.4 Estructuras de Control
### 2.5 Funciones

---

## 2.1 Python: Concepto, Variables y Asignación

### 🎯 **Teoría**

**¿Qué es Python?**
Python es un lenguaje de programación de alto nivel, interpretado y orientado a objetos. Fue creado por Guido van Rossum en 1991. Sus características principales incluyen:

- **Sintaxis simple y legible**: Similar al inglés natural
- **Tipado dinámico**: No necesitas declarar el tipo de variable
- **Interpretado**: Se ejecuta línea por línea
- **Multiplataforma**: Funciona en Windows, Mac, Linux
- **Gran ecosistema**: Miles de librerías para ciencia de datos

**Variables en Python**
Una variable es un contenedor que almacena datos en memoria. En Python:
- No necesitas declarar el tipo
- Se crean al asignar un valor
- Son referencias a objetos en memoria

### 💡 **Ejemplos**

```python
# Asignación básica de variables
nombre = "María"
edad = 25
altura = 1.75
es_estudiante = True

# Múltiples asignaciones
x, y, z = 1, 2, 3
a = b = c = 0

# Nombres de variables válidos
mi_variable = 10
variable123 = "texto"
_variable_privada = 42

# Nombres inválidos (comentar para evitar error)
# 2variable = 10  # No puede empezar con número
# mi-variable = 10  # No puede tener guiones
# class = "texto"   # No puede ser palabra reservada
```

### 🧪 **Ejercicios Prácticos**

**Ejercicio 1: Crear variables básicas**
```python
# Crea variables para almacenar:
# - Tu nombre completo
# - Tu edad
# - Tu altura en metros
# - Si te gusta la programación (True/False)
# Luego imprime toda la información

nombre_completo = "Tu nombre aquí"
edad = 0
altura = 0.0
gusta_programacion = True

print(f"Nombre: {nombre_completo}")
print(f"Edad: {edad} años")
print(f"Altura: {altura} metros")
print(f"¿Te gusta programar? {gusta_programacion}")
```

**Ejercicio 2: Intercambio de variables**
```python
# Intercambia los valores de dos variables sin usar una tercera variable
a = 10
b = 20

print(f"Antes: a = {a}, b = {b}")

# Tu código aquí para intercambiar valores
a, b = b, a

print(f"Después: a = {a}, b = {b}")
```

---

## 2.2 Tipos de Datos en Python

### 🎯 **Teoría**

Python tiene varios tipos de datos fundamentales:

**Tipos Básicos:**
- `int`: Números enteros (positivos, negativos, cero)
- `float`: Números decimales
- `str`: Cadenas de texto
- `bool`: Valores booleanos (True/False)

**Tipos Compuestos:**
- `list`: Listas ordenadas y mutables
- `tuple`: Tuplas ordenadas e inmutables
- `dict`: Diccionarios (clave-valor)
- `set`: Conjuntos únicos y desordenados

**Funciones útiles:**
- `type()`: Muestra el tipo de dato
- `isinstance()`: Verifica si un objeto es de cierto tipo

### 💡 **Ejemplos**

```python
# Tipos básicos
entero = 42
decimal = 3.14159
texto = "Hola mundo"
booleano = True

print(f"Tipo de {entero}: {type(entero)}")
print(f"Tipo de {decimal}: {type(decimal)}")
print(f"Tipo de {texto}: {type(texto)}")
print(f"Tipo de {booleano}: {type(booleano)}")

# Conversión de tipos
numero_texto = "123"
numero_convertido = int(numero_texto)
print(f"Convertido: {numero_convertido}, Tipo: {type(numero_convertido)}")

# Tipos compuestos
mi_lista = [1, 2, 3, "python"]
mi_tupla = (1, 2, 3)
mi_dict = {"nombre": "Ana", "edad": 25}
mi_set = {1, 2, 3, 3, 4}  # El 3 se elimina por duplicado

print(f"Lista: {mi_lista}")
print(f"Tupla: {mi_tupla}")
print(f"Dict: {mi_dict}")
print(f"Set: {mi_set}")
```

### 🧪 **Ejercicios Prácticos**

**Ejercicio 1: Identificar tipos de datos**
```python
# Identifica el tipo de cada variable y explica por qué
datos = [
    42,
    3.14,
    "Python",
    True,
    [1, 2, 3],
    (1, 2, 3),
    {"a": 1, "b": 2},
    {1, 2, 3}
]

for dato in datos:
    print(f"Valor: {dato} -> Tipo: {type(dato).__name__}")
```

**Ejercicio 2: Conversiones de tipo**
```python
# Convierte los siguientes datos al tipo especificado
texto_numero = "42"
decimal_texto = "3.14"
lista_texto = "[1, 2, 3]"

# Convierte texto_numero a entero
numero_entero = int(texto_numero)

# Convierte decimal_texto a float
numero_decimal = float(decimal_texto)

# Convierte lista_texto a lista (investiga eval())
lista_convertida = eval(lista_texto)

print(f"Entero: {numero_entero}, Tipo: {type(numero_entero)}")
print(f"Decimal: {numero_decimal}, Tipo: {type(numero_decimal)}")
print(f"Lista: {lista_convertida}, Tipo: {type(lista_convertida)}")
```

**Ejercicio 3: Crear estructuras de datos**
```python
# Crea las siguientes estructuras de datos:
# 1. Una lista con 5 frutas
# 2. Una tupla con coordenadas (x, y, z)
# 3. Un diccionario con información de un libro
# 4. Un set con colores únicos

frutas = ["manzana", "banana", "naranja", "uva", "pera"]
coordenadas = (10, 20, 30)
libro = {
    "titulo": "Python para Ciencia de Datos",
    "autor": "John Smith",
    "año": 2023,
    "paginas": 400
}
colores = {"rojo", "azul", "verde", "amarillo", "rojo"}  # El rojo se elimina

print("Frutas:", frutas)
print("Coordenadas:", coordenadas)
print("Libro:", libro)
print("Colores:", colores)
```

---

## 2.3 Operadores, Objetos y Punteros en Python

### 🎯 **Teoría**

**Operadores en Python:**

**Operadores Aritméticos:**
- `+` : Suma
- `-` : Resta
- `*` : Multiplicación
- `/` : División (siempre devuelve float)
- `//` : División entera
- `%` : Módulo (resto)
- `**` : Potencia

**Operadores de Comparación:**
- `==` : Igual a
- `!=` : Diferente de
- `<` : Menor que
- `>` : Mayor que
- `<=` : Menor o igual
- `>=` : Mayor o igual

**Operadores Lógicos:**
- `and` : Y lógico
- `or` : O lógico
- `not` : NO lógico

**Objetos y Referencias:**
En Python, todo es un objeto. Las variables son referencias (punteros) a objetos en memoria.

### 💡 **Ejemplos**

```python
# Operadores aritméticos
a = 10
b = 3

print(f"Suma: {a + b}")
print(f"Resta: {a - b}")
print(f"Multiplicación: {a * b}")
print(f"División: {a / b}")
print(f"División entera: {a // b}")
print(f"Módulo: {a % b}")
print(f"Potencia: {a ** b}")

# Operadores de comparación
x = 5
y = 10

print(f"{x} == {y}: {x == y}")
print(f"{x} != {y}: {x != y}")
print(f"{x} < {y}: {x < y}")
print(f"{x} > {y}: {x > y}")

# Operadores lógicos
es_mayor = True
es_estudiante = False

print(f"Es mayor Y estudiante: {es_mayor and es_estudiante}")
print(f"Es mayor O estudiante: {es_mayor or es_estudiante}")
print(f"NO es estudiante: {not es_estudiante}")

# Referencias de objetos
lista1 = [1, 2, 3]
lista2 = lista1  # Misma referencia

print(f"lista1: {lista1}")
print(f"lista2: {lista2}")
print(f"¿Son el mismo objeto? {lista1 is lista2}")

lista2.append(4)
print(f"Después de modificar lista2:")
print(f"lista1: {lista1}")
print(f"lista2: {lista2}")
```

### 🧪 **Ejercicios Prácticos**

**Ejercicio 1: Calculadora básica**
```python
# Crea una calculadora que realice las 4 operaciones básicas
num1 = 15
num2 = 4

suma = num1 + num2
resta = num1 - num2
multiplicacion = num1 * num2
division = num1 / num2

print(f"{num1} + {num2} = {suma}")
print(f"{num1} - {num2} = {resta}")
print(f"{num1} * {num2} = {multiplicacion}")
print(f"{num1} / {num2} = {division}")
```

**Ejercicio 2: Verificar condiciones**
```python
# Verifica si un número es par, positivo y mayor que 10
numero = 16

es_par = numero % 2 == 0
es_positivo = numero > 0
es_mayor_10 = numero > 10

cumple_todas = es_par and es_positivo and es_mayor_10

print(f"Número: {numero}")
print(f"¿Es par? {es_par}")
print(f"¿Es positivo? {es_positivo}")
print(f"¿Es mayor que 10? {es_mayor_10}")
print(f"¿Cumple todas las condiciones? {cumple_todas}")
```

**Ejercicio 3: Referencias y copias**
```python
# Demuestra la diferencia entre referencias y copias
original = [1, 2, 3]

# Referencia (mismo objeto)
referencia = original

# Copia (objeto diferente)
copia = original.copy()

print("Original:", original)
print("Referencia:", referencia)
print("Copia:", copia)

# Modificar la referencia
referencia.append(4)
print("\nDespués de modificar referencia:")
print("Original:", original)
print("Referencia:", referencia)
print("Copia:", copia)

# Modificar la copia
copia.append(5)
print("\nDespués de modificar copia:")
print("Original:", original)
print("Referencia:", referencia)
print("Copia:", copia)
```

---

## 2.4 Estructuras de Control

### 🎯 **Teoría**

Las estructuras de control permiten alterar el flujo de ejecución del programa:

**Condicionales:**
- `if`: Ejecuta código si la condición es verdadera
- `elif`: Condición adicional si la anterior es falsa
- `else`: Ejecuta código si ninguna condición es verdadera

**Bucles:**
- `for`: Itera sobre una secuencia (lista, tupla, string, etc.)
- `while`: Repite mientras la condición sea verdadera
- `break`: Sale del bucle inmediatamente
- `continue`: Salta a la siguiente iteración

### 💡 **Ejemplos**

```python
# Estructuras condicionales
edad = 18

if edad < 13:
    print("Eres un niño")
elif edad < 18:
    print("Eres un adolescente")
elif edad < 65:
    print("Eres un adulto")
else:
    print("Eres un adulto mayor")

# Bucle for
frutas = ["manzana", "banana", "naranja"]

for fruta in frutas:
    print(f"Me gusta la {fruta}")

# Bucle for con range
for i in range(5):
    print(f"Número: {i}")

# Bucle while
contador = 0
while contador < 3:
    print(f"Contador: {contador}")
    contador += 1

# Break y continue
for i in range(10):
    if i == 3:
        continue  # Salta el 3
    if i == 7:
        break     # Para en el 7
    print(i)
```

### 🧪 **Ejercicios Prácticos**

**Ejercicio 1: Sistema de calificaciones**
```python
# Crea un sistema que asigne letras según la calificación
def asignar_letra(calificacion):
    if calificacion >= 90:
        return "A"
    elif calificacion >= 80:
        return "B"
    elif calificacion >= 70:
        return "C"
    elif calificacion >= 60:
        return "D"
    else:
        return "F"

# Prueba con diferentes calificaciones
calificaciones = [95, 85, 75, 65, 55]

for cal in calificaciones:
    letra = asignar_letra(cal)
    print(f"Calificación {cal} = {letra}")
```

**Ejercicio 2: Contador de vocales**
```python
# Cuenta las vocales en una palabra
palabra = "programacion"
vocales = "aeiou"
contador = 0

for letra in palabra:
    if letra in vocales:
        contador += 1

print(f"La palabra '{palabra}' tiene {contador} vocales")
```

**Ejercicio 3: Números pares e impares**
```python
# Separa números pares e impares
numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pares = []
impares = []

for numero in numeros:
    if numero % 2 == 0:
        pares.append(numero)
    else:
        impares.append(numero)

print(f"Números pares: {pares}")
print(f"Números impares: {impares}")
```

**Ejercicio 4: Búsqueda en lista**
```python
# Busca un elemento en una lista
lista = ["python", "java", "javascript", "c++"]
buscar = "python"
encontrado = False

for elemento in lista:
    if elemento == buscar:
        encontrado = True
        break

if encontrado:
    print(f"'{buscar}' está en la lista")
else:
    print(f"'{buscar}' no está en la lista")
```

---

## 2.5 Funciones

### 🎯 **Teoría**

Las funciones son bloques de código reutilizables que realizan una tarea específica:

**Características:**
- **Reutilización**: Evita repetir código
- **Modularidad**: Divide el programa en partes manejables
- **Abstracción**: Oculta la complejidad
- **Mantenibilidad**: Facilita cambios y correcciones

**Sintaxis:**
```python
def nombre_funcion(parametros):
    # Código de la función
    return valor
```

**Tipos de parámetros:**
- **Posicionales**: Se pasan en orden
- **Con nombre**: Se especifica el nombre del parámetro
- **Por defecto**: Tienen un valor predeterminado
- **Arbitrarios**: `*args` (tupla) y `**kwargs` (diccionario)

### 💡 **Ejemplos**

```python
# Función básica
def saludar(nombre):
    return f"¡Hola {nombre}!"

# Función con parámetro por defecto
def saludar_con_titulo(nombre, titulo="Sr."):
    return f"¡Hola {titulo} {nombre}!"

# Función con múltiples parámetros
def calcular_area(base, altura):
    return base * altura

# Función con parámetros arbitrarios
def sumar_todos(*numeros):
    return sum(numeros)

# Función con parámetros con nombre
def crear_persona(nombre, edad, ciudad="Desconocida"):
    return {
        "nombre": nombre,
        "edad": edad,
        "ciudad": ciudad
    }

# Llamadas a funciones
print(saludar("Ana"))
print(saludar_con_titulo("Juan", "Dr."))
print(saludar_con_titulo("María"))  # Usa el valor por defecto
print(f"Área: {calcular_area(5, 3)}")
print(f"Suma: {sumar_todos(1, 2, 3, 4, 5)}")

persona = crear_persona(edad=25, nombre="Carlos")
print(f"Persona: {persona}")
```

### 🧪 **Ejercicios Prácticos**

**Ejercicio 1: Calculadora de estadísticas**
```python
def calcular_estadisticas(numeros):
    """Calcula estadísticas básicas de una lista de números"""
    if not numeros:
        return None
    
    total = sum(numeros)
    promedio = total / len(numeros)
    maximo = max(numeros)
    minimo = min(numeros)
    
    return {
        "total": total,
        "promedio": promedio,
        "maximo": maximo,
        "minimo": minimo,
        "cantidad": len(numeros)
    }

# Prueba la función
datos = [10, 20, 30, 40, 50]
stats = calcular_estadisticas(datos)

if stats:
    print("Estadísticas:")
    for key, value in stats.items():
        print(f"{key}: {value}")
```

**Ejercicio 2: Validador de contraseña**
```python
def validar_contraseña(contraseña):
    """Valida si una contraseña cumple con los requisitos"""
    errores = []
    
    if len(contraseña) < 8:
        errores.append("Debe tener al menos 8 caracteres")
    
    if not any(c.isupper() for c in contraseña):
        errores.append("Debe tener al menos una mayúscula")
    
    if not any(c.islower() for c in contraseña):
        errores.append("Debe tener al menos una minúscula")
    
    if not any(c.isdigit() for c in contraseña):
        errores.append("Debe tener al menos un número")
    
    return len(errores) == 0, errores

# Prueba diferentes contraseñas
contraseñas = ["abc", "password", "Password123", "MySecurePass1"]

for pwd in contraseñas:
    es_valida, errores = validar_contraseña(pwd)
    print(f"Contraseña: '{pwd}'")
    print(f"¿Es válida? {es_valida}")
    if not es_valida:
        print(f"Errores: {errores}")
    print()
```

**Ejercicio 3: Generador de secuencias**
```python
def generar_fibonacci(n):
    """Genera los primeros n números de la secuencia Fibonacci"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    secuencia = [0, 1]
    for i in range(2, n):
        siguiente = secuencia[i-1] + secuencia[i-2]
        secuencia.append(siguiente)
    
    return secuencia

def generar_pares(n):
    """Genera los primeros n números pares"""
    return [i * 2 for i in range(n)]

# Prueba las funciones
print("Fibonacci (10):", generar_fibonacci(10))
print("Pares (8):", generar_pares(8))
```

**Ejercicio 4: Función con múltiples retornos**
```python
def analizar_texto(texto):
    """Analiza un texto y retorna estadísticas"""
    if not texto:
        return 0, 0, 0, 0
    
    palabras = texto.split()
    caracteres = len(texto)
    caracteres_sin_espacios = len(texto.replace(" ", ""))
    oraciones = texto.count('.') + texto.count('!') + texto.count('?')
    
    return len(palabras), caracteres, caracteres_sin_espacios, oraciones

# Prueba la función
texto_ejemplo = "¡Hola mundo! Este es un ejemplo de texto. ¿Te gusta Python?"
palabras, chars, chars_sin_esp, oraciones = analizar_texto(texto_ejemplo)

print(f"Texto: '{texto_ejemplo}'")
print(f"Palabras: {palabras}")
print(f"Caracteres totales: {chars}")
print(f"Caracteres sin espacios: {chars_sin_esp}")
print(f"Oraciones: {oraciones}")
```

---

## 🏁 Desafío Final: Sistema de Gestión de Alumnos

Crea un sistema completo que combine todos los conceptos aprendidos:

```python
class Alumno:
    def __init__(self, nombre, edad):
        self.nombre = nombre
        self.edad = edad
        self.calificaciones = []
    
    def agregar_calificacion(self, calificacion):
        """Agrega una calificación a la lista"""
        if 0 <= calificacion <= 100:
            self.calificaciones.append(calificacion)
            return True
        return False
    
    def calcular_promedio(self):
        """Calcula el promedio de calificaciones"""
        if not self.calificaciones:
            return 0
        return sum(self.calificaciones) / len(self.calificaciones)
    
    def obtener_estado(self):
        """Determina el estado académico del alumno"""
        promedio = self.calcular_promedio()
        if promedio >= 90:
            return "Excelente"
        elif promedio >= 80:
            return "Bueno"
        elif promedio >= 70:
            return "Regular"
        else:
            return "Necesita mejorar"
    
    def __str__(self):
        return f"Alumno: {self.nombre}, Edad: {self.edad}, Promedio: {self.calcular_promedio():.2f}"

# Función para crear y gestionar alumnos
def gestionar_alumnos():
    alumnos = []
    
    # Crear algunos alumnos de ejemplo
    alumno1 = Alumno("Ana García", 20)
    alumno1.agregar_calificacion(95)
    alumno1.agregar_calificacion(88)
    alumno1.agregar_calificacion(92)
    
    alumno2 = Alumno("Carlos López", 19)
    alumno2.agregar_calificacion(75)
    alumno2.agregar_calificacion(82)
    alumno2.agregar_calificacion(78)
    
    alumnos.extend([alumno1, alumno2])
    
    # Mostrar información de todos los alumnos
    print("=== SISTEMA DE GESTIÓN DE ALUMNOS ===\n")
    
    for alumno in alumnos:
        print(f"Nombre: {alumno.nombre}")
        print(f"Edad: {alumno.edad}")
        print(f"Calificaciones: {alumno.calificaciones}")
        print(f"Promedio: {alumno.calcular_promedio():.2f}")
        print(f"Estado: {alumno.obtener_estado()}")
        print("-" * 40)
    
    # Estadísticas generales
    promedios = [alumno.calcular_promedio() for alumno in alumnos]
    promedio_general = sum(promedios) / len(promedios)
    
    print(f"Promedio general de la clase: {promedio_general:.2f}")

# Ejecutar el sistema
if __name__ == "__main__":
    gestionar_alumnos()
```

### 🎯 **Ejercicios Adicionales para Practicar**

1. **Extender el sistema de alumnos:**
   - Agregar más campos (email, carrera, etc.)
   - Implementar búsqueda por nombre
   - Agregar funcionalidad para eliminar calificaciones

2. **Crear una calculadora científica:**
   - Implementar funciones trigonométricas
   - Agregar operaciones con números complejos
   - Crear un menú interactivo

3. **Sistema de inventario:**
   - Crear clase Producto con stock, precio, categoría
   - Implementar funciones para agregar/quitar productos
   - Calcular valor total del inventario

---

## 📖 Recursos Adicionales

- [Documentación oficial de Python](https://docs.python.org/3/)
- [Tutorial de Python](https://docs.python.org/3/tutorial/)
- [Python para Ciencia de Datos](https://pandas.pydata.org/docs/)

---

*¡Recuerda practicar todos los conceptos y experimentar con el código!*