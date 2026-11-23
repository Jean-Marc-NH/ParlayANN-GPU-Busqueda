# Guion de la presentación

Búsqueda de ParlayANN en GPU — Computación Paralela, UCSP

- Duración: unos 17 minutos, a razón de un minuto por diapositiva más o menos.
- Está dividido en tres bloques por si nos repartimos la exposición. Cada bloque dura entre 5 y 6 minutos.
- Lo que va en *cursiva* son indicaciones (qué señalar o qué hacer), no se lee.
- Los números en negrita son los que conviene decir sí o sí.

---

## Bloque 1 — Qué hicimos y cómo funciona (portada a 05)

### Portada (20 s)

Buenas, somos Samuel Iman, Juan Victor Tejeda y Jean Marc Nadeau. En la presentación anterior
reprodujimos ParlayANN en CPU. Esta vez llevamos la parte de búsqueda a la GPU con CUDA y la
comparamos con la CPU en dos laptops distintas.

### 01 — Qué propone ParlayANN y qué hicimos nosotros (1 min)

*Señalar primero la columna izquierda y después la derecha.*

ParlayANN es un paper de PPoPP 2024. Propone construir índices de búsqueda aproximada de vecinos
cercanos usando grafos, en paralelo, sin usar locks y de forma determinista: si se corre dos veces,
sale exactamente lo mismo. Todo eso en CPU multinúcleo.

Nosotros no tocamos la construcción. Tomamos los mismos grafos que genera ParlayANN y programamos
en CUDA la búsqueda de los cuatro algoritmos. Lo primero que comprobamos es que la GPU devuelve los
mismos vecinos que la CPU, y recién después medimos tiempos en dos laptops.

*Leer el recuadro.* La idea es esta: si los dos devuelven exactamente lo mismo, la diferencia de
tiempo es solo del hardware y no de un algoritmo distinto o menos preciso.

### 02 — Cómo se busca en un grafo (1 min 15 s)

*Seguir con el dedo el camino azul del dibujo.*

Todos los algoritmos buscan igual. Se empieza en el punto 0 con una frontera de L_s candidatos. Se
toma el más cercano a la consulta que todavía no se visitó, se leen sus vecinos y se descartan los
que ya se vieron. Se calculan las distancias de los nuevos, se mezclan con la frontera, se recorta
a L_s y se repite hasta que ya no queda nada por visitar.

Para la GPU importan dos cosas. Primero, cada consulta es independiente, así que podemos resolver
miles a la vez. Segundo, casi todo el tiempo se va en leer vectores de memoria y calcular
distancias, y eso es lo que más se puede paralelizar.

*Leer el recuadro.* L_s es la perilla: más grande da más recall, pero menos consultas por segundo.

### 03 — Los cuatro algoritmos y sus grafos (1 min)

Los cuatro algoritmos se diferencian en cómo construyen el grafo. DiskANN inserta por lotes y poda,
HCNNG une árboles de expansión mínima, PyNNDescent refina con vecinos de vecinos y HNSW arma un
grafo en varios niveles. La búsqueda es la misma para los tres primeros; HNSW tiene la suya.

Usamos el dataset MSSPACEV de **un millón de vectores** de 100 dimensiones, con **29 316
consultas**, y pedimos los **10 vecinos** más cercanos. Es el mismo de la presentación anterior.

Un detalle: el grafo de HCNNG tiene casi 27 millones de aristas repetidas. Parece un error, pero
la CPU las recorre así, y si las quitábamos los resultados ya no coincidían.

### 04 — La búsqueda en GPU: un warp por consulta (1 min 15 s)

Cada consulta la resuelve un warp, que son 32 hilos. Cada warp toma siempre las mismas consultas,
así no necesitamos operaciones atómicas y el resultado no cambia de una corrida a otra. La frontera
y los candidatos están en memoria compartida, que es la más rápida.

*Señalar el dibujo.* Los vectores tienen 100 bytes y los rellenamos a 128 para leerlos de una sola
vez. Los 32 hilos se agrupan de a 8: cada grupo lee un vector y cada hilo lee 16 bytes. Además cada
hilo lleva 4 distancias, así que el warp pide **16 vectores a la vez**. Lo hacemos para que, mientras
llega un dato de memoria, ya haya otros pedidos en camino.

La distancia se calcula con `__dp4a`, una instrucción que multiplica y suma bytes en enteros. Sin
decimales, el número sale idéntico al de la CPU.

### 05 — Para que salga lo mismo que en la CPU (1 min 15 s)

*Columna izquierda.* Para que la GPU dé los mismos vecinos tuvimos que copiar detalles de ParlayANN
que no salen en el paper. Por ejemplo, el filtro que marca qué nodos ya se vieron tiene pérdidas: un
vecino puede pisar la casilla de otro y entonces se vuelve a revisar. También hubo que tratar la
frontera como multiconjunto por las aristas repetidas. En HNSW los candidatos se ordenan solo por
distancia y los vecinos se aceptan uno por uno, en orden.

*Columna derecha.* El problema es que la CPU hace todo eso vecino por vecino, y la GPU quiere hacerlo
todo a la vez. Lo resolvimos con `__match_any_sync`: los 32 hilos leen sus casillas al mismo tiempo
y cada uno sabe qué había justo antes de él, como si lo hubieran hecho en orden. En HNSW solo la
decisión de quién entra va en orden; leer, filtrar y calcular distancias sigue en paralelo.

*Leer el recuadro.* No fue por perfeccionismo: sin estos detalles el recall dejaba de coincidir.

---

## Bloque 2 — Optimización, hardware y validación (06 a 09)

### 06 — Optimizaciones del kernel (1 min)

*Recorrer las barras de izquierda a derecha.*

La gris es la CPU con 12 hilos: unas **32 mil consultas por segundo**. Nuestra primera versión usaba
un hilo por distancia y apenas llegaba a **1.34 veces** la CPU. Leer cada vector con 8 hilos ayudó un
poco, pero el salto grande vino al pedir toda la fila de vecinos de una vez: ahí llegamos a **141 mil
consultas por segundo, 4.4 veces** la CPU.

La barra roja es una idea que descartamos. En lugar de limpiar el filtro de visitados en cada
consulta, probamos marcarlo con un número de "época". Parecía más eficiente, pero salió un **30 %
más lento**, así que volvimos a limpiar.

Todo esto es en la RTX 3050 con DiskANN y L_s = 75.

### 07 — Hardware y forma de medir (1 min)

Usamos dos laptops. La máquina A tiene un i5 de 12.ª generación con 12 hilos y una RTX 3050. La
máquina B tiene un i7 de 13.ª generación con 24 hilos y una RTX 5050, que es Blackwell, la
arquitectura más nueva.

En cada máquina corrimos los 4 algoritmos con 18 configuraciones: **72 mediciones**. Hicimos una
corrida de calentamiento y nos quedamos con la mejor de 5, igual que ParlayANN. En la GPU medimos de
punta a punta, contando la copia de las consultas y la vuelta de los resultados, no solo el kernel.
La CPU usó todos sus hilos y las dos laptops estaban en modo de máximo rendimiento.

*Leer el recuadro.* Las máquinas no son equivalentes, por eso comparamos CPU contra GPU dentro de
cada una y aparte comparamos generaciones.

### 08 — ¿La GPU devuelve lo mismo que la CPU? (1 min)

Esta es la diapositiva más importante. En los cuatro algoritmos el recall es idéntico en las 18
configuraciones, en las dos máquinas. En DiskANN, HCNNG y PyNNDescent no solo coincide el recall:
para cada una de las 29 mil consultas, los 10 vecinos son exactamente los mismos.

En HNSW coincide el **99.4 %** de los vecinos. Revisamos el resto y en todos los casos son vecinos a
la misma distancia, o sea empates, que la CPU resuelve en otro orden. Por eso la columna de "mismas
distancias" está al **100 %** en todos.

Además todas las corridas son deterministas: la GPU da lo mismo en el calentamiento y en la última
corrida.

### 09 — Speedup con recall ≥ 0.95 (1 min)

Para comparar justo, fijamos un recall de 0.95, que es lo que se usa en la práctica, y tomamos la
configuración más rápida que lo cumple.

*Señalar las barras azules y luego las naranjas.* En la máquina A la GPU es entre **3.7 y 7.2 veces**
más rápida. En la máquina B, entre **6.6 y 10.7 veces**. La GPU ganó en las 72 configuraciones, y
copiar los datos entre CPU y GPU cuesta menos del 2 % del tiempo.

---

## Bloque 3 — Resultados, límites y cierre (10 a 15)

### 10 — ¿Qué rendimiento obtuvimos a alto recall? (1 min 15 s)

Esta tabla tiene los números detrás de la gráfica anterior. Por ejemplo, DiskANN: en la máquina A
pasa de 32 mil a 140 mil consultas por segundo, y en la B de 52 mil a **561 mil**, más de diez veces.

Las dos últimas columnas comparan las máquinas. *Leer el recuadro.* En DiskANN, HCNNG y PyNNDescent
la CPU nueva es entre 1.4 y 1.6 veces más rápida, aunque tiene el doble de hilos. La GPU nueva, en
cambio, es entre 3.4 y 4 veces más rápida. Por eso el speedup pasa de unas 4 veces a unas 10.

HNSW es la excepción: la CPU mejoró más (2.25 veces) que la GPU (2.07). En la máquina A tiene el
mayor speedup solo porque su versión de CPU es la más lenta.

### 11 — Curvas recall – QPS (45 s)

*Explicar los ejes antes de opinar.* En el eje horizontal está el recall y en el vertical las
consultas por segundo, en escala logarítmica. Las líneas punteadas son la CPU y las continuas la GPU.

La GPU está por encima en todo el rango, no solo en el punto de 0.95. Y en la máquina B la distancia
entre las dos curvas es mayor, que es lo mismo que vimos en la tabla.

### 12 — ¿Qué limita al kernel? (1 min 15 s)

*Izquierda.* Hicimos un microbenchmark que solo calcula distancias, para saber el techo de cada
GPU. La variante con 8 hilos por vector es la que usamos.

En la RTX 3050 nuestro kernel calcula **499 millones de distancias por segundo**, el **90 %** del
techo. O sea que ahí casi no hay más que ganar: lo que frena es el acceso a memoria.

*Derecha.* En la RTX 5050 el techo es casi 5 veces mayor y las distancias bajan a solo el 28 % del
tiempo. Empiezan a pesar ordenar los candidatos, unirlos con la frontera y hacer el corte, que son
partes más secuenciales dentro del warp. Ahí está lo que se podría optimizar después.

### 13 — Conclusiones de la comparación (1 min)

*Columna izquierda.* Lo que comprobamos: la GPU devuelve lo mismo que la CPU en las 72
configuraciones, y con recall de 0.95 es entre 3.7 y 10.7 veces más rápida. En tres de los cuatro
algoritmos la GPU mejoró más que la CPU entre generaciones. HNSW escala menos porque decidir qué
vecino entra es secuencial.

*Columna derecha.* Las limitaciones: usamos un solo dataset de un millón de puntos y laptops, donde
influyen la temperatura y el modo de energía. Además, HCNNG, PyNNDescent y HNSW quedaron con grafos
algo distintos en cada máquina; por eso para comparar generaciones el caso más limpio es DiskANN, que
salió idéntico. Y la construcción del grafo sigue en CPU.

*Leer el recuadro.*

### 14 — Trabajo futuro (45 s)

Lo siguiente es llevar la construcción del grafo a la GPU, empezando por DiskANN y HNSW, porque
insertan por lotes y comparten casi todo el código. Habría que medir el tiempo, la calidad del grafo
y si se mantiene el determinismo. También queremos probar con 10 millones de puntos o más, mientras
quepan en la memoria de la GPU, y reducir la parte secuencial de HNSW.

### 15 — Referencias y materiales (15 s)

Estas son las referencias principales. El código y todos los resultados están en el repositorio de
GitHub, por si quieren revisarlos o correrlos.

Eso es todo, gracias.

---

## Preguntas que probablemente nos hagan

**¿Por qué no usaron CAGRA o FAISS directamente?**
Porque construyen sus propios grafos. Nosotros queríamos comparar sobre el mismo grafo para que la
única diferencia sea el hardware.

**¿Por qué un warp por consulta y no un bloque?**
La frontera de una consulta cabe en memoria compartida y un warp se sincroniza sin costo. Con un
bloque habría que sincronizar entre warps en cada paso del beam search, que son muchos y cortos.

**¿El speedup incluye la copia de datos?**
Sí, la GPU se mide de punta a punta: copiar consultas, kernel y traer resultados. La copia pesa menos
del 2 %. La base de datos y el grafo se cargan una sola vez antes, igual que en la CPU.

**¿Por qué HNSW no coincide al 100 % en vecinos?**
Son empates: vecinos distintos a la misma distancia exacta. El recall es idéntico y las distancias
coinciden al 100 %.

**¿Cuánta memoria de GPU usa?**
Para un millón de puntos entran sin problema en los 6 GB de la 3050. El límite para escalar es la
VRAM: con datasets mucho más grandes habría que partir los datos o comprimirlos.

**¿Por qué la CPU nueva mejoró tan poco con el doble de hilos?**
El doble de hilos no significa el doble de núcleos rápidos: el i7 tiene 8 núcleos de rendimiento y 8
de eficiencia, y la mitad de sus hilos son de hyperthreading, que rinden menos que un núcleo
completo. Además la búsqueda depende del acceso a memoria, y la memoria no creció en la misma
proporción.
