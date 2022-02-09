# one-class-segmentation

## Versões:

- main.py: a média das features da ultima iteração do treino é usada para predizer. Sem nenhum tipo de detach.
- main_v2.py: igual a main.py porém com detach na hora de selecionar a, p, n
- main_knn.py: Sem detach. Predição com kNN.
- main_track_mean.py: Média atualizada ao longo do treino.

## Atualizações possiveis:

- [ ] parar de otimizar o prototype a partir de um ponto
- [ ] otimizar margin junto com a rede??
- [ ] otimizar o prototype com base somente nas classes positivas??
- [ ] treinar o prototype usando so a classe positiva e a rede com as duas classes???
- [ ] treinar o negativo contra todas as positivas (primeiro validar que a loss consegue agrupar todas as positivas bem pertinho)

## Material interessante:

- https://omoindrot.github.io/triplet-loss
