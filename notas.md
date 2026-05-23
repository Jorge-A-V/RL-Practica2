Cartpole Optimización

- 500 pasos consistentes

En github existe esta configuración https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml

Basicamente lo que hace son combinar varios factores que de por si deberían inferir en mejoras

- Por un lado la red es [256,256] en vez de [64,64] por lo que ya de por sí va a poder aproximar mejor la Q-function (como muchas veces en el deep learning, mas grande mejor o matar moscas a cañonazos)

- Cambia las actualizaciones de 4x1 a 256x128 de esta manera se acumulan bloques de 128 pasos cada 256 timesteps (mas o menos como un batch training)

- Realiza un intevalo de updates muy pequeño, (se actualiza la red casi constantemente (10) para asi evitar sinks de pasos)

- La exploración está muy concentrada, la epsilon de exploración disminulle de 16% a 4% de manera que el agente tiene un comportamiento bastante exlpotativo.