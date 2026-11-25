# Apresentação da etapa 6

PDF: `apresentacao.pdf`. Fonte editável: `slides.tex` (LaTeX). O roteiro abaixo distribui os 8 minutos entre os seis integrantes. Os tempos são metas; a equipe precisa ensaiar e cronometrar antes da aula.

| Integrante | Slide | Tempo | Fala principal |
|---|---:|---:|---|
| Alana Luize Kunrath | 1 | 1:00 | Problema, prioridade de leitura e limite: apoio à triagem, sem diagnóstico. |
| Murilo Ghedin Ghizzi | 2 | 1:30 | NIH ChestX-ray14, classes, 5.606 imagens e separação por paciente. |
| Mauro Junior Cella Schuastz | 3 | 1:30 | 150 imagens pareadas, cinco degradações e limitação da alternativa B. |
| Lucas Henrique Bonissoni | 4 | 1:30 | ResNet-50, limiares de decisão e checagem de qualidade. |
| Matheus de Aquino da Silva | 5 | 1:30 | Métricas no teste, falsos negativos e queda da balanced accuracy no degradado. |
| Luiz Augusto Cenci | 6 | 1:00 | Grad-CAM, erros selecionados, limitações e próximos passos. |

**Total planejado:** 8:00. Reservar 2:00 para perguntas. A equivalência das imagens sinteticamente degradadas às imagens próprias da opção B depende de concordância do professor.

## Perguntas prováveis

- **Por que não usar acurácia como métrica principal?** (Matheus) Porque a tarefa prioriza encontrar casos alterados. Recall e falsos negativos mostram a sensibilidade; balanced accuracy mostra o custo dos falsos positivos. No teste, recall foi 0,904, com 51 falsos negativos e 403 falsos positivos.
- **Como foi evitado vazamento entre treino e teste?** (Murilo) Os conjuntos foram separados por paciente. Uma imagem de um paciente do teste não aparece no treino ou na validação.
- **O que o Grad-CAM mostrou nos falsos negativos?** (Luiz) Nos três falsos negativos selecionados do teste, a maior ativação estava fora dos pulmões. É uma inspeção limitada, não prova causalidade nem generaliza para todos os erros.
- **Por que não há imagens próprias reais?** (Mauro) Foi adotada a alternativa B planejada para imagens médicas: 150 exames públicos do teste receberam degradações sintéticas pareadas. A aceitação para a disciplina deve ser confirmada com o professor.
- **O modelo pode ser usado clinicamente?** (Alana) Não. O conjunto é público, os rótulos não foram revisados clinicamente e não houve validação externa ou prospectiva.
- **Por que há uma faixa indeterminada?** (Lucas) Para separar probabilidades próximas do limiar de rotina, onde uma decisão binária direta seria frágil. A faixa tem largura de 0,05 para cada lado do limiar.
