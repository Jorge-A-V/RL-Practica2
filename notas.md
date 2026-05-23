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