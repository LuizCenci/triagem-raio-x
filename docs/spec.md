## UTFPR • ENGENHARIA DE SOFTWARE Visão Computacional

## Trabalho Final da Disciplina

Especificação

## Do dataset público às suas próprias imagens

Cada equipe parte de um dataset público, acrescenta imagens próprias e conta o que aconteceu em um artigo curto e em uma apresentação.

Professor

Marlon Marcon

Modalidade

Trabalho em equipe, no tema já divulgado e escolhido

Entregas

Artigo curto (50 %) + apresentação (50 %)

Data

26/11 — entrega do artigo e apresentação

Curso de Engenharia de Software • Disciplina de Visão Computacional


## 1 O trabalho em resumo

## Quatro pontos

- 1. Escolher um dataset público adequado ao tema da equipe.

- 2. Acrescentar imagens próprias a esse dataset — para ajudar no treinamento ou para testar a transferência de conhecimento do modelo.

- 3. Escrever um artigo curto (short paper) — 50% da nota.

- 4. Apresentar o trabalho e os resultados — 50% da nota.

Entrega do artigo e apresentação: 26/11.

O objetivo é que a equipe passe pela experiência central de um projeto de visão computaci- onal: usar dados que já existem, colocar dados seus no problema, treinar um modelo e relatar com honestidade o que funcionou e o que não funcionou. Desempenho baixo não reprova — o que se avalia é a clareza do que foi feito e a qualidade da análise.

## 2 O que fazer

## 2.1 Escolher um dataset público

Cada equipe trabalha no tema que já escolheu, entre os temas divulgados. Os datasets sugeridos na divulgação dos temas são um bom ponto de partida, mas a equipe pode usar outro, desde que seja público e adequado ao problema. No artigo, informe de onde ele vem (nome e link), a licença, quantas imagens tem e quais são as classes.

Não é preciso usar o dataset inteiro: um recorte (menos classes, menos imagens) é perfeita- mente aceitável, desde que seja descrito.

## 2.2 Acrescentar imagens próprias

A equipe deve acrescentar ao dataset algumas imagens que não fazem parte dele:

fotos ou vídeos feitos pelos integrantes, de preferência no contexto local (campus, cidade, casa, laboratório). As imagens precisam ser anotadas no formato da tarefa — rótulo da classe, caixas ou máscaras. Há duas formas de usá-las; escolha uma (ou as duas, se quiser):

|   | Opção A — no treinamento | Opção B — no teste (transferên- |
| --- | --- | --- |
|   |   | cia) |
| Como usar | As imagens próprias entram no | O modelo é treinado só com o da- |
|   | treino, junto com o dataset pú- | taset público e testado nas imagens |
|   | blico. | próprias. |
| Pergunta | Acrescentar as minhas imagens | O que o modelo aprendeu no da- |
|   | melhorou o modelo? | taset público vale para as minhas |
|   |   | imagens? |
| O que comparar | Resultado sem e com as imagens | Resultado no teste do dataset pú- |
|   | próprias no treino. | blico e nas imagens próprias. |

Não há quantidade mínima rígida. Como referência, 30 a 50 imagens bem escolhidas já permitem uma comparação que faz sentido; prefira poucas imagens variadas (ângulo, iluminação, fundo) a muitas imagens iguais. Na opção A, guarde uma parte das imagens próprias fora do treino para poder medir o efeito.


## Dois cuidados

Imagem de teste não entra no treino. Se uma imagem foi usada para treinar ou para

ajustar o modelo, ela não serve para medir o resultado.

Privacidade. Fotos com rostos, placas de veículos ou dados pessoais só com consentimento

de quem aparece, e com essas informações borradas ou tarjadas antes de guardar ou publicar.

Ferramentas de anotação gratuitas e sem instalação: makesense.ai, Roboflow, CVAT e Label [URL 🔗](https://www.makesense.ai/)

Studio. Quando não for possível fotografar (imagens médicas ou de satélite, por exemplo), combine uma alternativa com o professor. [URL 🔗](https://labelstud.io/)

## 2.3 Treinar e avaliar um modelo

Treine um modelo para a tarefa do tema partindo de uma rede pré-treinada (transfer lear-

ning), com as ferramentas vistas em aula — TensorFlow/Keras ou Ultralytics. Os notebooks da pasta codigos/ são pontos de partida prontos, e o Google Colab gratuito é suficiente: prefira modelos leves (MobileNetV2, YOLOv8n, U-Net compacta) e treinos curtos.

Avalie com uma métrica adequada à tarefa (por exemplo, acurácia e F1 para classificação,

mAP para detecção, IoU para segmentação) e mostre também exemplos visuais de acertos e de erros — principalmente nas imagens próprias.

## 3 Entregas e avaliação

| Entrega | O que é |   | Quando Valor |
| --- | --- | --- | --- |
| Artigo curto | PDF de 4 a 6 páginas, no modelo da SBC |   | 26/11, até as 19h 50% |
| Apresentação | 8 minutos + 2 de perguntas, em aula | 26/11 | 50% |

## 3.1 Artigo curto (short paper) — 50%

| Formato | Modelo de artigos da SBC, entregue em PDF |
| --- | --- |
| Extensão | 4 a 6 páginas, incluindo figuras e referências |
| Idioma | Português |
| Prazo | 26/11, até as 19h (antes das apresentações) |

## Estrutura sugerida:

- 1. Introdução — o problema e por que ele importa.

- 2. Dados — o dataset público e as imagens próprias: como foram obtidas, anotadas e usadas (opção A ou B).

- 3. Método — o modelo, como foi treinado e como foi avaliado.

- 4. Resultados e discussão — os números, a comparação da opção escolhida, exemplos de acertos e erros, limitações.

- 5. Conclusão — o que a equipe aprendeu e o que faria a seguir.

- 6. Referências.

Inclua no artigo o link do código (notebook do Colab ou repositório) e das imagens próprias.

## 3.2 Apresentação — 50%

Cada equipe apresenta em aula, no dia 26/11: 8 minutos de apresentação e 2 minutos de

perguntas. O formato dos slides é livre. Um roteiro que funciona: o problema; o dataset público


e as imagens próprias (mostre as imagens); o modelo; os resultados, com exemplos de acertos e erros; o que a equipe aprendeu.

Todos os integrantes devem participar da apresentação e estar prontos para responder às

perguntas.

## 3.3 Critérios de avaliação

## 4 Regras gerais

- Nota. A nota é da equipe. O professor pode diferenciar a nota da apresentação de quem não participar dela ou não souber explicar o próprio trabalho.

- Prazo. O artigo deve ser entregue até as 19h de 26/11. A equipe que não apresentar nessa data fica sem a nota da apresentação, salvo justificativa aceita pelo professor.

- IA generativa. O uso de assistentes de IA é permitido e deve ser declarado em uma frase no artigo (qual ferramenta, para quê). Não entregue texto, código ou resultado que a equipe não saiba explicar.

- Integridade. Plágio ou resultado fabricado implica nota zero.

## 5 Perguntas frequentes

Não reprova. Um resultado modesto, bem explicado, vale mais do que um número alto sem

análise. Na opção B, é normal o modelo ir pior nas imagens próprias — explicar essa diferença é justamente o resultado do trabalho.

Não há mínimo rígido; use 30 a 50 imagens como referência. Em vídeo, valem os quadros anotados.

Parta de pesos pré-treinados, mas adapte o modelo ao seu dataset — é isso que permite a

comparação das opções A e B.

Use o dataset público mais próximo do seu problema e trate as suas imagens como o teste de

## E se o desempenho ficar ruim?

## Quantas imagens próprias precisamos?

## Podemos usar um modelo pronto, sem treinar nada?

## Meu tema não tem um dataset público pronto. E agora?


transferência (opção B). Em caso de dúvida, fale com o professor.

## Precisamos de GPU própria?

Não. O Colab gratuito é suficiente.

## 6 Lista de verificação

- Dataset público identificado no artigo (nome, link, licença)

- Imagens próprias coletadas, anotadas e descritas no artigo

- Opção A ou B aplicada, com a comparação correspondente

- Métrica adequada à tarefa e exemplos de acertos e erros

- Artigo em PDF, modelo SBC, 4 a 6 páginas

- Link do código e das imagens próprias no artigo

- Uso de IA generativa declarado

- Artigo entregue até as 19h de 26/11

- Slides prontos, ensaiados em 8 minutos, com todos os integrantes participando
