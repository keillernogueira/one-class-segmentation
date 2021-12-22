# one-class-segmentation

## Versões:

- main.py: a média das features da ultima iteração do treino é usada para predizer. Sem nenhum tipo de detach.
- main_v2.py: igual a main.py porém com detach na hora de selecionar a, p, n
- main_knn.py: Sem detach. Predição com kNN.
- main_track_mean.py: Média atualizada ao longo do treino.

## Atualizações possiveis:

- [X] atualizar o mean ao longo do treino (main_track_mean.py)
- [X] predizer com knn (main_knn.py)
- [ ] usar somente as amostras positivas mais longe do mean
- [ ] usar a tecnica que selecionar amostras mendeley 
- [ ] testar prototype networks
- [ ] testar o triplet loss sem o mean

## Material interessante:

- https://omoindrot.github.io/triplet-loss
