# Roteiro da apresentação (8 minutos)

Ordem sorteada em 25/11/2026: **Lucas Bonissoni, Luiz Cenci, Mauro Cella, Matheus de Aquino, Murilo Ghizzi e Alana Kunrath**. Cada pessoa apresenta dois slides em 1:20 (aproximadamente 0:40 por slide). O roteiro é uma proposta de fala, não uma transcrição obrigatória. Ensaiem com cronômetro e preservem 2 minutos para perguntas.

| Pessoa | Slides | Janela |
|---|---:|---:|
| Lucas Bonissoni | 1–2 | 0:00–1:20 |
| Luiz Cenci | 3–4 | 1:20–2:40 |
| Mauro Cella | 5–6 | 2:40–4:00 |
| Matheus de Aquino | 7–8 | 4:00–5:20 |
| Murilo Ghizzi | 9–10 | 5:20–6:40 |
| Alana Kunrath | 11–12 | 6:40–8:00 |

## Lucas Bonissoni

**Slide 1, 0:00–0:40.** "Boa noite. Nosso projeto investiga uma forma de organizar a fila de leitura de radiografias de tórax. A ideia é destacar exames com maior probabilidade de alteração para que um profissional possa analisá-los mais cedo. Trabalhamos com uma amostra pública de 5.606 imagens do NIH ChestX-ray14, divididas em NORMAL e ALTERADA. Isso é um experimento acadêmico de apoio à triagem. O modelo não determina a doença e não substitui um radiologista."

**Slide 2, 0:40–1:20.** "O fluxo começa com a radiografia, passa pela ResNet-50 e produz uma probabilidade para a classe ALTERADA. A camada de decisão transforma essa probabilidade em prioridade alta, média, rotina ou indeterminado. Também há uma verificação simples de qualidade da imagem. Nosso resultado, portanto, é uma proposta de ordem de leitura, sujeita à revisão humana. Luiz vai explicar de onde vieram os dados e como evitamos misturar pacientes entre os conjuntos."

## Luiz Cenci

**Slide 3, 1:20–2:00.** "Usamos uma amostra do NIH ChestX-ray14. A classe NORMAL corresponde ao rótulo original No Finding; os demais achados foram agrupados como ALTERADA. Essa simplificação é importante: não classificamos doenças específicas. Também precisamos lembrar que os rótulos foram extraídos de laudos e não receberam uma revisão clínica no nosso projeto. Por isso, os resultados devem ser interpretados dentro desse conjunto público, e não como desempenho clínico já comprovado."

**Slide 4, 2:00–2:40.** "Dividimos as 5.606 imagens por paciente: 3.931 para treino, 531 para validação e 1.144 para teste. Um mesmo paciente não aparece em mais de uma divisão, o que reduz vazamento de informação. A validação foi usada para ajustar limiares e limites de qualidade. O teste ficou separado para a avaliação final. Esses números refletem somente o recorte que usamos. Mauro vai mostrar a arquitetura e a regra que converte a previsão em prioridade."

## Mauro Cella

**Slide 5, 2:40–3:20.** "A principal rede é a ResNet-50 com pesos pré-treinados no ImageNet. Redimensionamos as radiografias para 224 por 224 pixels e adaptamos os tons de cinza para três canais. O treino teve duas fases: primeiro a base ficou congelada; depois ajustamos parte do bloco final. Usamos ponderação de classes e aumentos de dados. Para saber se a transferência de aprendizado ajudou, comparamos essa rede com uma CNN simples treinada do zero."

**Slide 6, 3:20–4:00.** "A rede retorna uma probabilidade de ALTERADA. O limiar de rotina, 0,231, foi escolhido para favorecer recall, aceitando mais falsos positivos. Valores próximos desse limiar entram em uma faixa indeterminada de 0,05 para cada lado; prioridade alta começa em 0,788. Antes disso, a checagem de qualidade compara contraste e nitidez com limites da validação. Essa checagem é limitada e pode deixar passar algumas falhas. Matheus vai mostrar o que ocorreu no teste."

## Matheus de Aquino

**Slide 7, 4:00–4:40.** "Nas 1.144 imagens do teste, a ResNet-50 alcançou recall de 90,4%, F1 de 67,8% e AUROC de 72,5%. O baseline teve 85,5% de recall, 61,8% de F1 e 53,7% de AUROC. Também melhoraram precisão e balanced accuracy. Destacamos recall porque queremos evitar que exames alterados fiquem sem atenção antecipada. Porém, uma métrica alta sozinha não mostra todos os erros; a matriz de confusão dá esse contexto."

**Slide 8, 4:40–5:20.** "No teste, houve 479 verdadeiros positivos, 51 falsos negativos, 403 falsos positivos e 211 verdadeiros negativos. Os falsos positivos aumentam a carga da fila, e os falsos negativos são especialmente preocupantes. Na camada de prioridade, 44 exames foram para alta, 725 para média, 188 para indeterminado e 187 para rotina. Desses 187 em rotina, 34 tinham rótulo ALTERADA. Por isso apresentamos o sistema como apoio à triagem, não como uma decisão segura para uso autônomo. Murilo vai avaliar a robustez."

## Murilo Ghizzi

**Slide 9, 5:20–6:00.** "Para investigar a sensibilidade à qualidade da imagem, selecionamos 150 radiografias do teste, metade NORMAL e metade ALTERADA. Criamos uma versão degradada de cada uma: ruído, rotação, corte, baixo contraste ou uma combinação, com 30 imagens em cada grupo. Assim comparamos cada original com a sua versão alterada usando o mesmo limiar. São transformações sintéticas, não exames novos coletados em ambiente clínico."

**Slide 10, 6:00–6:40.** "Nas 150 imagens pareadas, a balanced accuracy caiu de 0,600 para 0,520 e o F1 de 0,688 para 0,654. O recall subiu de 0,880 para 0,907, mas isso veio com mais falsos positivos: de 51 para 65; a precisão caiu. Portanto, o aumento do recall não representa melhora geral. A checagem de qualidade detectou todas as imagens de baixo contraste e corte nessa amostra, mas nenhuma de rotação ou ruído. Alana encerra com a inspeção visual dos erros e os próximos passos."

## Alana Kunrath

**Slide 11, 6:40–7:20.** "Inspecionamos 15 mapas Grad-CAM, sendo 12 casos do teste e três degradados. Nos três falsos negativos selecionados do teste, a ativação principal ficou fora dos pulmões. Isso chama atenção para possíveis atalhos visuais usados pelo modelo. Mas o Grad-CAM não prova a causa da decisão, e esses poucos exemplos não representam todos os erros. A inspeção ajuda a formular hipóteses para avaliações futuras."

**Slide 12, 7:20–8:00.** "Concluímos que a ResNet-50 melhorou as métricas em relação ao baseline, mas ainda produziu 51 falsos negativos e 403 falsos positivos no teste. A avaliação com degradações mostrou perda de balanced accuracy e lacunas na checagem de qualidade. Antes de qualquer uso clínico seriam necessários dados externos, rótulos revisados, melhor calibração e testes do impacto na fila real. Para a disciplina, a aceitação das imagens sinteticamente degradadas como alternativa B ainda precisa da concordância do professor. Obrigada; estamos prontos para as perguntas."

## Respostas para perguntas prováveis

- **Por que não priorizar acurácia?** A triagem exige atenção aos falsos negativos; por isso reportamos recall, F1, balanced accuracy, AUROC e a matriz de confusão. A acurácia isolada esconderia a troca entre sensibilidade e falsos positivos.
- **Houve vazamento de dados?** A separação foi por paciente. Os limiares foram ajustados na validação, e o teste foi usado para avaliação final.
- **O que significa o Grad-CAM fora dos pulmões?** É uma observação nos três falsos negativos selecionados do teste, não uma explicação causal nem uma conclusão sobre todos os exames.
- **Por que imagens degradadas em vez de imagens próprias?** É uma alternativa sintética planejada para o contexto de imagens médicas. A aceitação pela disciplina depende da concordância do professor.
- **O modelo pode ser usado em hospital?** Não. Faltam validação externa, revisão clínica dos rótulos, avaliação prospectiva e estudo do fluxo de trabalho.
