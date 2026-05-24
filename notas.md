## 4.1 Cartpole Optimización

- 500 pasos consistentes

En github existe esta configuración https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml

Basicamente lo que hace son combinar varios factores que de por si deberían inferir en mejoras

- Por un lado la red es [256,256] en vez de [64,64] por lo que ya de por sí va a poder aproximar mejor la Q-function (como muchas veces en el deep learning, mas grande mejor o matar moscas a cañonazos)

- Cambia las actualizaciones de 4x1 a 256x128 de esta manera se acumulan bloques de 128 pasos cada 256 timesteps (mas o menos como un batch training)

- Realiza un intevalo de updates muy pequeño, (se actualiza la red casi constantemente (10) para asi evitar sinks de pasos)

- La exploración está muy concentrada, la epsilon de exploración disminulle de 16% a 4% de manera que el agente tiene un comportamiento bastante exlpotativo.

## 4.2 Lunarlanding Discreto

Opcion 1 de la pŕactica, LunarLander-v3 con opciones de movimientos discretos de BOx2D, donde una nave tuene que aterrizar en una paltaforma marcada con dos banderas

Acciones: La nave tiene Discrete(4) acciones discretas (nada, motor izquierdo derecho y princial)

Observaciones (Box(8,)) que sonn la posicion (x,y) la velocidad (vx,vy) el ángulo, la velocidad angular y dos flags de contacto con el suelo

El reward penaliza en base a la distancia con el obbjetivo, el consumo de combustile, esttrellarse es un -100, un aterrizaje  suave es un +100 y un resuelto es un reward promedio de 200

### Algorimos

Se van a probar los siguietnes algorimtos

DQN: Como algoritmo referencia que usa el replay buffer para mejorar las actuacioes de cada vez

PP0: es una opcion mas o menos modernilla, se sigue usando (al menos yo conozco qgente que la usa). Debería ser bastante estable y bastante robusto con hipers y funciona tanto discreto como Box (MlpPolicy en este caso)

A2C: Por probar algo ligero y mas o menos rápido, aun lo escuche hace poco en un reunión.

# Hiperparametros

Igual que del abnteior podemos sacar hiperparamentros bastante correctos de: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams (segun los comentarios indica el nivel de tuneado que tienen) Usaremos esos como base pero realizamos un mini CrossValidation de variaciones por probar.

Bueno crossvalidation no que es RL (me lo dijeron en una reunion del TFM) vamos a variar un poco el learning rate y las seeds para probar variaciones simulando el minimo cross validation

---

Se escogen los tres algoritmos para asi poder comparar un off-policy con un on-policy (actor crítico) y un actor crítico más sencillo

#### DQN

Es Q learning pero con Q table sustituida por una red neuronal

Las transiciones (s,a,r,s) se guardan y se muestrean en minibatchs aleatoris. Rompe asi la correlacion temporal y permite reutilizar experiencias.

La Target network es lo que solo se actualiza cada update_interval y asi se soluciona cambiar el objetivo cada vez que se entrena le red

Tiene un factor de exploracion

Tiene sentido dentro del problema porque la operacion max_Q sobre 4 valores discretos es simple y no requiere de trucos, de la misma forma, unn mlp pequeña sirve para cubrir las observaciones de nuevo discretas. Un reward denso favorece a las actualizaciones de DQN.

#### PPO

Con PPO se hacen n pasos con la politica actual, se estima la ventaja con la estimacion general (GAE) suvaizando la varianza (lo gcontrola el gae_lamda). La poplitica se actualiza haciendo un pequeño cliping para proteger el entorno y eso le da la estabilidad. Se entrena un critico aparte con MSE.

Se puede usar poqeu soporta valores discretos y tiene naturaleza estocástica sin neceisdad del parametro de exploracion. PPO a mayores funciona en principio bastante bien out of the box por lo que no requiere de mucho ajuste de hiperparametros.

Es más sensible a la cantidad de steps, tamaño de batch y el numero de epocas ya que suele requerir una cantidad de pasos más o menos considerables.

#### A2C (Advantage actor critic)

Se tiene un actor y un crítico. Cada N steps se calcula una ventaja y se actualiza. El actor tiene un gradiente estándar y un pequeño bonus de entropia y el critico es un MSE con los entornos calculados (no tiene ni clipiing ni nada de eso (mas rapido pero menos fino))

Sirve para comparar con el PPO (mas bruto) y es relativamente rápido para probar. Es suficientemente simple para el problema de 4-8 como se tiene con el lunarlanding.

Es mas simple más ruidoso y menos eficiente que el ppo, puede ser algo sensible al lr si por la entropía.

---

Con estos 3 modelos podemos curbir valores vs politica, complejidad y velocidad en las técnicas.

---

El ZOO para los hiperpartamentso es la mejor refeencia posible. Cambiamos un par de cosillas por simplicidad. En el mini cv no se usa multilples entornos por la complejidad computacionel y en el A2C se cambia el LR lineal decreciennte a uno constante para que tenga una naturaleza similar a las otras opciones (y para mantener y reutilizar la extructura del mini cv)

En principio DQN si que está optimmizado para LunarLanding, los otros no necesariamente (50/50)

## Resuldaos


Los resultados finales son que DQN consigue 211 con 67 de sttdev en 8 minutos, PPO 273 con 22 de stdev en 17 y A=C 56 y -91 con 160/170 de varianza en 2.

En este caso PPO es el claro agnador, tiene mejor media con mucha menos desviacion dipica, pero sacrificas tiempo. DQN consigue resultados por encima de lo que se consideraba como threshold, pero con mas varianza.

A2C no converge para esta politica, a mayores de las pruebas que estan ahora se han probado diferentes estrategias y no da entrenado, la inestabilidad que se decia antes del A2C hace que no funcione

---

PPO es estadisticamente mejor que el resto.