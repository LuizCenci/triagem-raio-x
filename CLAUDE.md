UNIVERSIDADE TECNOLÓGICA FEDERAL DO PARANÁ

Campus Dois Vizinhos

Bacharelado em Engenharia de Software

## Trabalho de Visão Computacional - Entrega 1

Alana Luize Kunrath - alanaluize@alunos.utfpr.edu.br - RA 2523701 Murilo Ghedin Ghizzi - muriloghizzi@alunos.utfpr.edu.br - RA 2580608 Mauro Junior Cella Schuastz - mschuastz@alunos.utfpr.edu.br - RA 2580586 Lucas Henrique Bonissoni - bonissoni@alunos.utfpr.edu.br - RA 2554704 Matheus de Aquino da Silva - matheuss.2005@alunos.utfpr.edu.br - RA 2554747 [URL 🔗](mailto:matheuss.2005@alunos.utfpr.edu.br)

Luiz Augusto Cenci - luizcenci@alunos.utfpr.edu.br - 2554712 [URL 🔗](mailto:luizcenci@alunos.utfpr.edu.br)

## 1. Introdução e definição do problema

A análise de radiografias de tórax é uma etapa essencial do fluxo de atendimento em serviços de saúde, sobretudo quando o volume de exames é grande e os casos mais urgentes precisam ser vistos primeiro. Nesse cenário, um dos principais desafios é identificar com rapidez os exames que apresentam algum tipo de alteração, para que possam ser analisados com

prioridade na fila.

Este projeto propõe um sistema de Visão Computacional que classifica radiografias de tórax em incidência frontal em duas categorias: NORMAL, quando não há achados registrados, e ALTERADA, quando há ao menos um achado associado à imagem. A partir dessa classificação, o sistema estima uma probabilidade de alteração para cada exame, que serve de base para a priorização. A inteligência artificial é tratada pelo grupo como ferramenta de

apoio à triagem, e não como meio de identificar doenças específicas.

Cabe à Visão Computacional analisar automaticamente os padrões presentes nas radiografias e sinalizar os exames com maior chance de estarem alterados. Para isso, o trabalho emprega técnicas de aprendizado profundo e transferência de aprendizado, adaptando um modelo

previamente treinado ao problema de classificação de imagens médicas.

O sistema não se destina a diagnóstico. Conforme a própria proposta do trabalho, sua função

se limita a organizar a ordem em que os exames são analisados, servindo de apoio à priorização e nunca como substituto da avaliação feita por profissionais de saúde. Ao colocar à frente os exames com maior probabilidade de alteração, contribui para uma fila de atendimento mais bem organizada sem assumir o papel de diagnóstico clínico.

## 2. Objetivos do sistema e decisão-alvo

O sistema deve analisar radiografias frontais de tórax e estimar a probabilidade de cada exame apresentar alguma alteração. Esse resultado orienta a posição do exame na fila de análise, de acordo com o nível de prioridade atribuído a ele.


A classificação principal é binária. A classe NORMAL corresponde às imagens rotuladas como No Finding na base de dados, e a classe ALTERADA reúne as imagens que apresentam ao menos um dos demais achados disponíveis no conjunto utilizado. O sistema não busca identificar qual doença ou alteração está presente, e sim indicar se há chance de existir algum achado relevante na imagem.

A saída do modelo é uma probabilidade de alteração, 𝑝(𝑎𝑙𝑡𝑒𝑟𝑎𝑑𝑎) , que é repassada a uma camada de decisão responsável por convertê-la em uma recomendação de prioridade. As saídas possíveis do sistema são:

- IMAGEM INADEQUADA: a imagem não tem qualidade suficiente para análise, seja por rotação excessiva, corte, baixo contraste ou baixa nitidez;

- INDETERMINADO: o modelo apresenta alta incerteza, como quando a probabilidade fica próxima do limiar de decisão ou há evidências visuais contraditórias;

- PRIORIDADE ALTA: o exame tem alta probabilidade de alteração;

- PRIORIDADE MÉDIA: o exame tem probabilidade intermediária de alteração;

- ROTINA: o exame tem baixa probabilidade de alteração.

Os limiares que separam as faixas de prioridade são definidos no conjunto de validação, e não no de teste. A estratégia adotada privilegia a revocação (sensibilidade) da classe ALTERADA, de modo a reduzir o número de exames alterados encaminhados indevidamente para ROTINA.

A decisão produzida pelo sistema é, portanto, uma recomendação de priorização, e não um diagnóstico médico. O objetivo é apoiar a triagem para que os exames com maior chance de alteração sejam avaliados primeiro, mantendo a avaliação de todos os exames por um médico.

## 3. Base de dados pública

O projeto utiliza o conjunto NIH ChestX-ray14, disponibilizado pelo National Institutes of Health (NIH). A base reúne 112.120 radiografias de tórax de 30.805 pacientes e foi construída para apoiar pesquisas sobre classificação automática de achados radiográficos.

Como o conjunto é multirrótulo, o problema foi simplificado para uma classificação binária: imagens com o rótulo No Finding compõem a classe NORMAL, e imagens com ao menos um dos demais achados formam a classe ALTERADA. Essa simplificação é coerente com o objetivo do sistema, que não pretende identificar doenças específicas, e sim estimar se há ou não alguma alteração relevante.

A base também traz metadados como identificador do paciente, idade, sexo e incidência da radiografia (PA ou AP). Essas informações permitem separar os conjuntos por paciente e, posteriormente, analisar o desempenho do modelo em diferentes subgrupos.

O conjunto completo ocupa cerca de 42 GB, volume elevado para o escopo deste trabalho. Por isso, utilizamos uma amostra que mantém exemplos suficientes das classes NORMAL e ALTERADA e preserva a separação por paciente entre os conjuntos de treinamento, validação e teste.


Entre as vantagens do NIH ChestX-ray14 estão o grande número de imagens, a disponibilidade de informações por paciente e a proximidade com um cenário real de triagem. O identificador de paciente, em particular, reduz o risco de vazamento de dados entre treinamento e teste, preocupação central nesse tipo de aplicação.

A base também tem limitações. Os rótulos foram extraídos automaticamente dos laudos por técnicas de processamento de linguagem natural, o que introduz ruído de anotação; o catálogo da disciplina indica acurácia aproximada de 90% para esses rótulos, de modo que parte das classificações pode estar incorreta. Há ainda o risco de o modelo aprender atalhos visuais, como marcadores, dispositivos médicos ou características dos equipamentos de aquisição, em vez de se basear apenas em informações relacionadas aos pulmões. Por fim, o desempenho obtido nessa base não garante resultado equivalente em outras instituições ou populações, já que a origem dos dados e o perfil dos pacientes podem introduzir vieses. Essas limitações são consideradas na avaliação e na discussão dos resultados.

Quanto à licença, o catálogo da disciplina informa que o conjunto é de divulgação pública pelo NIH e pode ser acessado sem credenciamento, mas não especifica uma licença formal. Essa informação deve ser confirmada na fonte oficial antes da versão final do relatório.

## 4. Pré-processamento e estruturação dos dados

Para manter a consistência das imagens e evitar vieses no treinamento e na avaliação do modelo, os dados passam pelas etapas descritas a seguir.

Os rótulos originais do NIH ChestX-ray14 são agrupados em duas classes: imagens rotuladas como No Finding formam a classe NORMAL, e imagens com ao menos um achado formam a classe ALTERADA.

Antes do treinamento, as imagens são redimensionadas para a resolução de entrada exigida pela arquitetura, de modo que todos os dados tenham o mesmo formato. Os valores dos pixels são normalizados conforme o pré-processamento do modelo pré-treinado, garantindo compatibilidade com a abordagem de transfer learning.

Os arquivos seguem uma organização padronizada, com dados originais, processados e degradados em subpastas distintas, como na estrutura abaixo:

text

```
None
data/
├── raw/
├── processed/
├── train/
```


```
├── validation/
├── test/
└── degraded/
```

A divisão dos dados é feita por paciente, e não por imagem. O grupo definiu inicialmente a proporção de cerca de 70% dos pacientes para treinamento, 10% para validação e 20% para teste, com semente fixa para permitir a reprodução dos resultados.

Esse cuidado é necessário porque um mesmo paciente pode ter várias radiografias na base. Se imagens de um mesmo indivíduo aparecerem tanto no treinamento quanto no teste, o modelo pode aprender características específicas dessa pessoa e apresentar desempenho artificialmente alto, fenômeno conhecido como vazamento de dados, que compromete a validade da avaliação. O próprio catálogo do tema aponta a presença do mesmo paciente em treinamento e teste como uma das principais armadilhas desse problema.

Para evitá-lo, a separação utiliza os identificadores de paciente presentes nos metadados, com ferramentas como GroupShuffleSplit ou GroupKFold e o Patient ID como grupo. O código inclui ainda uma verificação automática que confirma a ausência de interseção de pacientes entre treinamento, validação e teste.

O conjunto de teste permanece isolado durante todo o desenvolvimento e só é usado na avaliação final. Limiares de decisão, hiperparâmetros e demais escolhas do modelo são definidos apenas com os conjuntos de treinamento e validação, o que reduz o risco de sobreajuste ao conjunto de teste.

As imagens do conjunto degradado são geradas exclusivamente a partir do conjunto de teste e mantêm seus rótulos originais. Isso permite comparar diretamente o desempenho do modelo nas imagens originais e nas versões degradadas, avaliando sua robustez em condições de aquisição menos favoráveis.

## 5. Conjunto de teste degradado

Para este projeto, não será realizada coleta local de novas radiografias, em razão da natureza médica das imagens e das limitações éticas e práticas associadas à obtenção desse tipo de dado. Em substituição à coleta local, será criado um conjunto de teste degradado a partir de imagens já presentes na base pública utilizada no projeto. Essa adaptação é prevista especificamente para este tema no catálogo da disciplina.

O conjunto degradado será composto por 150 imagens provenientes exclusivamente do conjunto de teste, preservando seus rótulos originais. A seleção deverá manter equilíbrio aproximado entre as classes, com cerca de 75 imagens NORMAL e 75 imagens ALTERADA, permitindo comparar o comportamento do modelo nas duas situações.


As imagens selecionadas serão submetidas a diferentes tipos de degradação, simulando condições de aquisição inadequadas ou perda de qualidade. A distribuição definida pelo grupo será:

- 30 imagens com ruído, utilizando técnicas como ruído gaussiano ou sal e pimenta;

- 30 imagens com rotação, em ângulos aproximados entre ±10° e ±30°;

- 30 imagens com corte, removendo entre 20% e 40% das bordas da imagem;

- 30 imagens com baixo contraste;

- 30 imagens com degradações combinadas, aplicando dois ou mais tipos de alteração sobre a mesma imagem.

A geração dessas imagens será realizada de forma reproduzível, utilizando uma semente fixa e registrando os parâmetros aplicados em cada transformação. Também será criado um arquivo degraded_metadata.csv, contendo informações como a imagem original, o tipo de degradação, os parâmetros utilizados e o respectivo rótulo.

O objetivo desse conjunto é avaliar a robustez do modelo diante de condições desfavoráveis de aquisição. O desempenho obtido nas imagens degradadas será comparado ao desempenho nas versões originais, permitindo medir a queda em métricas como Recall, F1-score e Balanced Accuracy. Dessa forma, será possível verificar se o sistema mantém capacidade adequada de triagem mesmo quando a qualidade da radiografia é reduzida.

Além da avaliação de desempenho, o conjunto degradado também poderá auxiliar na identificação de imagens que não apresentem qualidade suficiente para uma classificação confiável, contribuindo para a categoria IMAGEM INADEQUADA definida na camada de decisão do sistema.

## 6. Modelo proposto

A classificação das radiografias é feita por uma Rede Neural Convolucional (CNN), abordagem amplamente usada em Visão Computacional por aprender automaticamente as características relevantes das imagens.

O modelo principal é a ResNet-50 com pesos pré-treinados no ImageNet. A arquitetura foi escolhida por extrair características visuais complexas por meio de conexões residuais e por ser consolidada em tarefas de classificação de imagens. Como alternativa para comparação, pode ser usada uma arquitetura mais leve, como a MobileNetV2 ou a EfficientNet-B0.

A estratégia de treinamento se baseia em transfer learning e ocorre em duas etapas. Na primeira, a ResNet-50 atua como extratora de características, com o backbone convolucional congelado, e a camada de classificação original é substituída por uma nova cabeça de saída voltada ao problema binário, responsável por gerar a probabilidade de a radiografia pertencer à classe ALTERADA. Apenas essa nova cabeça é treinada nessa fase, o que aproveita as características visuais já aprendidas no pré-treinamento e dispensa o treinamento da rede inteira desde o início.

Na segunda etapa, pode ser realizado o fine-tuning, em que parte dos últimos blocos da ResNet-50 é descongelada e treinada com taxa de aprendizado reduzida. O objetivo é ajustar


as características mais específicas da rede ao domínio das radiografias de tórax sem descaracterizar o conhecimento adquirido no pré-treinamento.

Como as classes podem estar desbalanceadas, o treinamento considera técnicas de ponderação, como class weights ou o aumento do peso da classe positiva na função de perda. A escolha entre elas depende da análise da distribuição das imagens selecionadas para o projeto.

O desempenho do modelo principal é comparado a um baseline mais simples, que serve de referência para verificar se a arquitetura convolucional com transferência de aprendizado traz ganho efetivo em relação a uma abordagem menos elaborada.

Ao final, o modelo não fornece diagnóstico nem prioridade de atendimento. Sua saída é a probabilidade 𝑝(𝑎𝑙𝑡𝑒𝑟𝑎𝑑𝑎) , que a camada de decisão do sistema converte nas categorias de prioridade definidas no projeto.

## 7. Procedimento de avaliação

O modelo é avaliado com métricas adequadas a problemas de classificação binária e a bases potencialmente desbalanceadas. Como o objetivo do projeto é apoiar a triagem e reduzir o número de exames alterados encaminhados como rotina, a avaliação não se apoia apenas na acurácia global.

A métrica principal é o Recall (sensibilidade) da classe ALTERADA, isto é, a proporção de exames alterados que o modelo identifica corretamente. Ela ocupa papel central porque um Recall baixo significa mais falsos negativos, ou seja, exames alterados classificados como normais ou de baixa prioridade.

A Precisão mede a proporção de exames classificados como alterados que de fato pertencem à classe ALTERADA. Com ela, é possível avaliar a quantidade de falsos positivos e verificar se a priorização em excesso sobrecarrega a fila sem necessidade. O F1-score combina Precisão e Recall em uma única medida, equilibrando a capacidade de detectar exames alterados e a confiabilidade das classificações positivas.

A Balanced Accuracy considera separadamente o desempenho em cada classe, o que impede que a classe majoritária domine o resultado quando há desbalanceamento entre NORMAL e ALTERADA. O conteúdo da disciplina destaca que, nesse tipo de base, a acurácia convencional pode passar uma impressão enganosa do desempenho, e recomenda o uso de F1-score e Balanced Accuracy.

A matriz de confusão complementa as métricas numéricas, mostrando diretamente a quantidade de verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos. Ela é especialmente útil para identificar quantos exames da classe ALTERADA foram classificados incorretamente.

O grupo também analisa AUROC, PR-AUC e especificidade, além do Recall por faixa de prioridade e por subgrupos disponíveis nos metadados, como sexo, faixa etária e incidência da


radiografia. A queda de desempenho no conjunto degradado em relação às imagens originais também é medida.

Os limiares que convertem 𝑝(𝑎𝑙𝑡𝑒𝑟𝑎𝑑𝑎) nas categorias de prioridade são definidos exclusivamente no conjunto de validação. O conjunto de teste não participa dessa escolha e fica reservado para a avaliação final do sistema. A estratégia inicial busca alta sensibilidade para a classe ALTERADA, com foco em reduzir os exames alterados enviados para ROTINA.

A avaliação, portanto, considera o desempenho geral do classificador, mas dá peso maior à capacidade de identificar corretamente exames potencialmente alterados e de manter comportamento consistente nas diferentes condições testadas.

## 8. Critério de decisão e custo dos erros

Neste projeto, os erros do classificador não têm o mesmo impacto. Como o sistema apoia a priorização de radiografias na fila de análise, é preciso considerar de forma explícita o custo dos falsos positivos e dos falsos negativos.

O erro mais grave é o falso negativo, que ocorre quando uma radiografia da classe ALTERADA é classificada como normal ou enviada para uma faixa de baixa prioridade, como ROTINA. Nesse caso, um exame relevante pode demorar a ser analisado, o que reduz a eficácia da triagem. O próprio tema do trabalho destaca a necessidade de levar em conta essa assimetria entre os dois tipos de erro.

O falso positivo ocorre quando uma radiografia normal é classificada como alterada ou recebe prioridade maior do que a necessária. Esse erro aumenta a carga de trabalho dos profissionais e gera priorizações desnecessárias, mas seu impacto é menor, já que apenas antecipa a análise de um exame que seria avaliado de qualquer forma.

Por isso, a decisão é orientada a manter alta sensibilidade (Recall) da classe ALTERADA, com foco na redução dos falsos negativos. Os limiares que convertem 𝑝(𝑎𝑙𝑡𝑒𝑟𝑎𝑑𝑎) nas categorias de prioridade são ajustados no conjunto de validação com esse objetivo. A proposta inicial do grupo é que a revocação dos exames ALTERADOS encaminhados para PRIORIDADE ALTA ou PRIORIDADE MÉDIA fique acima de uma meta previamente definida.

A matriz de confusão e as demais métricas são interpretadas à luz dessa diferença de impacto. Uma redução moderada da especificidade pode ser aceita se trouxer uma queda expressiva nos falsos negativos, desde que o volume de falsos positivos continue compatível com o objetivo do sistema.

O sistema não retira nenhum exame da análise profissional; ele apenas reorganiza a ordem da fila, e mesmo os exames classificados como ROTINA continuam sendo avaliados. O principal risco de um falso negativo é, portanto, o atraso na análise, e não a exclusão do exame do fluxo de atendimento.


O critério de decisão segue, assim, uma política conservadora de triagem, que prioriza a identificação dos casos potencialmente alterados e reconhece que, nesta aplicação, um falso negativo custa mais do que um falso positivo.

## 9. Explicabilidade com Grad-CAM

Para complementar as métricas quantitativas, o projeto utiliza o Grad-CAM (Gradient-weighted Class Activation Mapping) na interpretação das decisões da rede. A técnica gera mapas de ativação que indicam as regiões da imagem que mais influenciaram a classificação.

Com esses mapas, é possível verificar se o modelo se concentra nas áreas relevantes da radiografia, sobretudo na região pulmonar, ou se recorre a elementos externos e características sem relação com o objetivo da classificação. O conteúdo da disciplina apresenta o Grad-CAM justamente como uma forma de visualizar quais regiões da imagem contribuíram para a decisão de uma CNN.

Essa análise é especialmente relevante no NIH ChestX-ray14, já que existe o risco de o modelo aprender atalhos visuais da base e usar textos, marcadores, bordas, dispositivos médicos ou características do equipamento de aquisição como pistas para classificar. O catálogo do tema aponta esse risco como uma das principais armadilhas do projeto.

O catálogo exige a aplicação do Grad-CAM em no mínimo cinco casos. Para ampliar a análise, o grupo pretende gerar cerca de 12 mapas, distribuídos entre os diferentes resultados do modelo: três verdadeiros positivos, três falsos negativos, três falsos positivos e três verdadeiros negativos, podendo incluir imagens do conjunto degradado.

Em cada caso, observa-se se as regiões destacadas se concentram nos pulmões ou se o modelo se apoia em áreas inadequadas, como bordas, textos, marcadores ou dispositivos. Essa inspeção permite identificar comportamentos suspeitos que passariam despercebidos apenas com as métricas de classificação.

Os falsos negativos recebem atenção especial, por serem os erros mais críticos para a triagem. Os mapas desses casos ajudam a entender se o modelo deixou de captar características importantes ou se baseou a decisão em regiões inadequadas da radiografia.

Os mapas do Grad-CAM são tratados como ferramenta de explicabilidade qualitativa, e não como prova de que o modelo raciocina como um profissional de saúde. Seu papel é reunir evidências sobre o comportamento do classificador e ajudar a revelar vieses ou atalhos aprendidos durante o treinamento.

## 10. Limitações, riscos e aspectos éticos

Um sistema de Visão Computacional aplicado a radiografias de tórax envolve limitações técnicas e questões éticas que precisam ser consideradas em todas as etapas do projeto. Por se


tratar de uma aplicação em saúde, os resultados do modelo não devem ser tomados como diagnóstico médico.

A única função do sistema é a triagem: a probabilidade de alteração serve para organizar a prioridade dos exames, enquanto a decisão clínica permanece sob responsabilidade de um profissional habilitado. Esse limite é central, já que o próprio tema da disciplina estabelece que a solução deve ser apresentada como apoio à triagem, e não como sistema de diagnóstico.

Uma das principais limitações está no conjunto de dados. No NIH ChestX-ray14, os rótulos foram extraídos automaticamente dos laudos por técnicas de processamento de linguagem natural, e o catálogo indica acurácia aproximada de 90%, o que revela a presença de ruído de anotação. Algumas imagens podem, assim, ter rótulos incorretos ou incompletos, afetando tanto o treinamento quanto a avaliação.

Há também possíveis vieses na base. O perfil dos pacientes, os equipamentos, as instituições de origem e as condições de aquisição podem não representar adequadamente outros contextos, e bases diferentes costumam variar em protocolo, resolução, população e critério de anotação. Um bom desempenho no conjunto utilizado não garante comportamento equivalente em imagens de outros hospitais, equipamentos ou populações, nem permite extrapolar os resultados automaticamente para outros cenários clínicos.

Outro risco relevante é o aprendizado de atalhos visuais. Em vez de se basear em características da região pulmonar, o modelo pode associar a classe a textos, marcadores, bordas, dispositivos médicos ou particularidades do equipamento de aquisição. Esse problema, apontado como uma das principais armadilhas do tema, é investigado por meio dos mapas gerados pelo Grad-CAM.

O possível desbalanceamento entre as classes também limita a interpretação dos resultados. Se uma categoria predominar, a acurácia pode passar uma visão enganosa do desempenho, razão pela qual o projeto adota métricas complementares, como Recall, F1-score e Balanced Accuracy, e estratégias de ponderação no treinamento quando necessário.

Quanto à privacidade e ao uso responsável das imagens, o projeto utiliza apenas dados de bases públicas destinadas à pesquisa, sem coleta local de radiografias de pacientes. Os dados não são usados para identificar indivíduos, e as informações da base são tratadas exclusivamente no escopo acadêmico do trabalho.

Mesmo que o modelo alcance métricas elevadas, seus resultados devem ser lidos à luz dessas limitações. O sistema não substitui a avaliação médica, não deve ser usado isoladamente em decisões clínicas e é apresentado como um protótipo acadêmico para investigar técnicas de Visão Computacional no apoio à triagem de radiografias.

## 11. Organização do repositório

O projeto é mantido e documentado no repositório cxr-triage-vc, no GitHub, que centraliza o código-fonte, os notebooks dos experimentos, a documentação, os resultados e o relatório da equipe. A estrutura inicial é a seguinte:


text

```
None
cxr-triage-vc/
├── data/
│ └── README.md
├── notebooks/
├── src/
├── results/
├── report/
├── README.md
└── requirements.txt
```

A pasta data/ documenta a organização dos dados do projeto. Como as imagens da base ocupam muito espaço, elas não são versionadas no GitHub, e o arquivo data/README.md traz as instruções para obter e organizar a base utilizada.

A pasta notebooks/ reúne os notebooks Jupyter das etapas de exploração dos dados, pré-processamento, treinamento e avaliação. Os arquivos seguem um padrão numérico e descritivo, como 01_eda.ipynb, 02_preprocessamento.ipynb e 03_treinamento.ipynb, que deixa clara a sequência dos experimentos.

A pasta src/ concentra o código reutilizável, com as funções de carregamento e preparação dos dados, treinamento dos modelos, avaliação, geração do conjunto degradado e implementação do Grad-CAM, de modo que as funcionalidades centrais não fiquem presas aos notebooks.

Em results/ ficam os resultados dos experimentos, como métricas, matrizes de confusão e demais arquivos gerados na avaliação. As configurações e os resultados de cada experimento podem ser registrados em JSON ou CSV, acompanhados de uma tabela com data, configuração, semente aleatória e métricas obtidas.

A pasta report/ guarda os documentos das entregas da disciplina, incluindo o relatório da primeira etapa e, mais adiante, a documentação final do projeto.

O README.md principal apresenta a visão geral do projeto, seus objetivos, as instruções de configuração do ambiente, a estrutura do repositório e as orientações para executar o código. O requirements.txt registra as bibliotecas e dependências utilizadas, permitindo reproduzir o ambiente de desenvolvimento.


No controle de versão, a branch main é protegida. Cada integrante trabalha em uma branch própria ou vinculada a uma tarefa específica e abre um Pull Request, que só é integrado à main após a aprovação de pelo menos outro membro da equipe.

Essa organização garante rastreabilidade, colaboração entre os integrantes e reprodutibilidade dos experimentos ao longo do desenvolvimento.

## 12. Repositório do projeto

O código-fonte, os notebooks, a documentação e os resultados desenvolvidos ao longo do projeto serão disponibilizados em um repositório no GitHub, permitindo o acompanhamento da evolução do trabalho e o acesso aos materiais produzidos pela equipe.

## Link do repositório:

O repositório será mantido atualizado durante o desenvolvimento do projeto e servirá como fonte principal para consulta do código, organização dos experimentos e documentação das entregas.

```
https://github.com/LuizCenci/triagem-raio-x
```
