# TODO

- [ ] Testar para mais estados
  - [ ] Florida
  - [ ] California
  - [ ] Texas

# Completed

- [x] Testar Hex2Vec com as configs do notebook do GeoVex
  - [x] Selecionar configuração alternativa do hex2vec
  - [x] Gerar embeddings
  - [x] Testar modelo e avaliar métricas
    - [x] Layer
      - [x] Normal Data
      - [x] Big Data
    - [x] View
      - [x] Normal Data
      - [x] Big Data
- [x] Geovex
  - [x] Gerar embeddings para os 3 estados (10, 50 e 100)
  - [x] Avaliar métricas do modelo
    - [x] Region View
    - [x] Region Layer
  - Big Data
    - [x] Region View
    - [x] Region Layer
- [x] Melhorar Execução da versão baseline
- [x] Testar com Big Data
  - [x] Hex2vec
    - [x] Region Layer
    - [x] Region view
- [x] Analisar se a abordagem usando region view, de fato, melhora o modelo
- [x] Implementar multihead attention
- [x] Executar modelo com multihead attention
- [x] Executar embeddings no PGCNN
  - Responder: utilizando o pgcnn conseguimos melhorar? se sim, conseguimos deixar o modelo melhor que o baseline do HAVANA?
  - [x] Region Layer
  - [x] Region View
- [x] Avaliação de Métricas PGCNN x HAVANA
  - [x] Definir melhor métricas para cada configuração de cada estado (lembrar de comparar no maximo 10 por vez, devido ao limite de visualização)
  - Para o PGCNN, utilizando uma region view, tivemos melhores métricas em relação ao baseline e à abordagem de region layer
  - Comparando as métricas de region view no PGCNN com o baseline do HAVANA, temos que o HAVANA performa melhor
  - Um ponto importante é que a visão de região performou melhor no PGCNN do que no HAVANA
- [x] Alterar max_size_matrix
  - [x] Testar com menor volume de dados
- [x] POI encoder
  1. get datasets -> utils
     - entrada: arquivo pbf ([geofrabrik.de](https://download.geofabrik.de/) -> north america -> sub regions -> usa -> estados)
  2. poi encoder
     - main
     - file_name pois -> pbf
     - boroughs -> boundary
     - vai gerar o poi encoder tension
  3. region embedding
