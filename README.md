# TOPSIS-Sort-C
O projeto TOPSIS-Sort-C engloba o desenvolvimento do algoritmo homônimo em Python, sua disponibilização como uma biblioteca Python acessível no PyPI (Python Package Index), e a criação de uma interface de teste amigável baseada na web utilizando o Streamlit. Este trabalho foi inspirado pelo artigo ["Sorting with TOPSIS through boundary and characteristic profiles"](https://doi.org/10.1016/j.cie.2020.106328) (Ordenando com TOPSIS através de perfis de limite e característica).

O TOPSIS-Sort-C expande os conceitos centrais do TOPSIS (Technique for Order Preference by Similarity to Ideal Solution), introduzindo uma abordagem refinada para lidar com perfis de limite e características. O TOPSIS é uma técnica de tomada de decisão multiatributo amplamente utilizada para classificar e selecionar as melhores alternativas com base em critérios pré-definidos.

## Experimente o TOPSIS-Sort-C

Para utilizar a lib do nosso algoritmo, acesse o [link](https://pypi.org/project/TOPSIS-Sort-C/). Além disso, disponibilizamos uma página no Streamlit para que seja possível testar no seu navegador, basta acessar o [link](https://topsis-sort.streamlit.app/).

-------

## Instalação

Para instalar a biblioteca basta utilizar:

```bash
pip install TOPSIS-Sort-C
```

------

## Exemplo de uso

### Entrada manual
```bash
matriz = [[8, 10, 6], [7, 6, 9], [10, 9, 9]]
```
```bash
perfis = [[10, 10, 9], [8, 7, 9], [7, 6, 6]]
```
```bash
pesos = [9, 8, 8]
```
```bash
criterios = [True, False, True]
```

### Arquivo .csv
Matriz:
```bash
8, 10, 6
7, 6, 9
10, 9, 9
```
Perfis:
```bash
10, 10, 9
8, 7, 9
7, 6, 6]
```
Pesos:
```bash
9, 8, 8
```
Critérios:
```bash
True, False, True
```

------

## Métodos

------

## Limitações

------
## Autores do projeto

Esse é o time de alunos, da graduação em Sistemas de Informações, responsável pelo desenvolvimento do projeto:

| [<img src="https://avatars.githubusercontent.com/u/53124770?v=4" width=115><br><sub>Geovanna Domingos - gmdn@cin.ufpe.br </sub>](https://github.com/geovannaadomingos) |  [<img src="https://avatars.githubusercontent.com/u/104395661?v=4" width=115><br><sub>Gustavo de Hollanda - ghcs@cin.ufpe.br </sub>](https://github.com/gustavo-ghcs) |  [<img src="https://avatars.githubusercontent.com/u/103337809?v=4" width=115><br><sub>Maria Eduarda Melo - meom@cin.ufpe.br </sub>](https://github.com/Madu218) |
| :---: | :---: | :---:
| [<img src="https://avatars.githubusercontent.com/u/116587792?v=4" width=115><br><sub>Higor Cunha - hbc@cin.ufpe.br</sub>](https://github.com/higorcunha1) |  [<img src="https://avatars.githubusercontent.com/u/86128256?v=4" width=115><br><sub>Giovanna Machado - gpmb@cin.ufpe.br</sub>](https://github.com/giovannamachado) |
