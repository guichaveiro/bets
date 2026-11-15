# Jogo patológico na APS e nos CAPS, 2018–2026

Este documento explica o que a análise faz e por quê. O código está em `analise.py`, a saída completa em `resultados.txt` e as figuras em `figuras/`. As seções têm a mesma numeração nos três arquivos.

```
uv run python artigo/analise.py          # uns 15 minutos; --B 99 para testar rápido
```

## 1. Dados

Usamos só dados públicos agregados. Não há ligação de pacientes, então não há submissão ao CEP (Res. CNS 510/2016, art. 1º).

Na atenção primária, a fonte é o relatório público de produção do SISAB: atendimentos individuais por mês, Brasil, de fev/2018 a set/2026. A unidade é o atendimento em que o CID foi registrado. Um mesmo atendimento pode ter mais de um CID.

Nos CAPS, a fonte são as RAAS do SIA-PS, baixadas do FTP do DATASUS, de jan/2018 a jun/2026. A unidade é a linha de RAAS, isto é, um procedimento registrado, datado pelo dia do atendimento. Do arquivo o script lê só cinco colunas (estabelecimento, data do atendimento, sexo, CID principal e data de início da RAAS). O cartão SUS e a data de nascimento não são lidos.

Três recortes de período:

- jul–ago/2026 ficam de fora nos CAPS porque ainda estão chegando;
- falta um arquivo dos 2.808 (AP, dez/2023);
- jan/2018 fica de fora na APS. É o primeiro mês do SISAB e tem 34 registros de jogo, contra 5 a 18 nos onze meses seguintes. Esse ponto sozinho criava uma quebra falsa no começo da série. A série completa aparece na robustez.

Os grupos de CID:

| Grupo | Códigos | Para quê |
|---|---|---|
| Jogo | F63.0 | desfecho |
| Referência | F32, F33, F41, F43, F10, F20 | representa o sistema: a demanda geral de saúde mental, usada como denominador |
| Controle negativo | F63.1, F63.2, F63.3 (piromania, cleptomania, tricotilomania) | mesmo agrupamento do jogo, mesmas regras de codificação, nada a ver com apostas |
| F63 sem especificação | F63, F63.8, F63.9 | não é controle, porque pode conter jogo; serve para ver se a alta é só codificação mais específica (só CAPS) |

Registro não é pessoa. Quem está em acompanhamento gera vários registros.

## 2. O modelo

Todos os testes usam regressão segmentada (joinpoint), a mesma ideia do programa Joinpoint do NCI, comum na CSP. Para contagens mensais:

```
log E[y_t] = log(denominador_t) + 11 dummies de mês + pandemia (mar/2020–dez/2021)
             + β·t + Σ δ_k·(t − τ_k)+
```

A tendência é uma linha quebrada. Em cada quebra τ a inclinação muda, mas a curva não salta.

- Contagens com denominador usam binomial negativa, com o denominador como offset. Quando o desfecho é uma fração (jogo / (jogo + controle)) usamos binomial com logit.
- As quebras são procuradas em todas as posições possíveis, com pelo menos 12 meses entre elas e das pontas. Testamos de 0 a 2 quebras e escolhemos pelo BIC. Entre modelos a menos de 2 pontos do melhor, fica o mais simples.
- O resultado de cada segmento é a variação percentual anual (VPA), 100·(e^(12·inclinação) − 1). A VPAM é a média do período todo (o que o Joinpoint chama de AAPC).

Sobre os intervalos de confiança: o principal é o do bootstrap. Ele reamostra blocos de 6 meses dos resíduos (999 réplicas) e, em cada réplica, procura as quebras de novo. Assim o intervalo carrega a incerteza sobre onde a quebra está. O número de quebras fica fixo nas réplicas. O intervalo HAC (Newey-West, 6 defasagens) também aparece, mas é condicional às quebras encontradas e por isso fica estreito demais.

A existência da quebra é decidida pelo BIC. O z da mudança de inclinação que aparece no `resultados.txt` não é teste disso: com a posição da quebra estimada, esse z não tem a distribuição usual. Só serve como medida de tamanho.

## 3. Descritiva

Tabela por ano com as contagens, a taxa de jogo e do controle por 10 mil registros de referência, a taxa por milhão de habitantes e a % de mulheres.

## 4. Pergunta principal: o registro de jogo cresceu mais que o sistema?

Primeiro medimos quanto o sistema cresceu, ajustando o modelo às próprias contagens de referência, sem denominador. Fazemos o mesmo com a referência ampliada com CIAP na APS e com o total de RAAS nos CAPS.

Depois vem o modelo principal: contagem de F63.0 com offset log(referência). Com o offset, o crescimento do sistema já sai descontado. O que sobra é a tendência do jogo por unidade de demanda de saúde mental.

A hipótese se sustenta se o BIC escolher pelo menos uma quebra e o intervalo bootstrap da VPA do último segmento ficar todo acima de zero.

## 5. Especificidade

A alta poderia ser só mudança no jeito de codificar: o e-SUS passando a usar CID, a RAAS sendo preenchida de outro jeito, a rede sendo treinada. Uma mudança dessas afetaria os outros códigos F63 também.

Por isso rodamos o mesmo modelo no controle (F63.1–F63.3). Se ele não sobe, ponto para o jogo.

O teste direto é a fração F63.0 / (F63.0 + F63.1–F63.3), em binomial com logit. A inclinação desse modelo é exatamente a diferença entre o crescimento do jogo e o do controle. Se a VPA do último segmento for positiva, o jogo cresce mais.

Nos CAPS olhamos também os F63 sem especificação. Eles podem esconder casos de jogo. Se crescem junto, parte da alta do F63.0 pode ser só gente trocando F63 por F63.0. Testamos isso de duas formas: somando esses códigos ao desfecho e usando-os como comparador no lugar do controle. No SISAB esses códigos não foram consultados.

## 6. Robustez

A VPA do último segmento é refeita em vários cenários. A tabela usa intervalo HAC, que é condicional às quebras.

| Cenário | O que testa |
|---|---|
| até 3 quebras; 1 quebra fixa | a escolha do número de quebras |
| sem os meses da pandemia | a distorção de 2020–2021 |
| série até dez/2025 | meses recentes ainda sendo consolidados |
| HAC com 12 defasagens | autocorrelação mais longa (só muda o intervalo) |
| APS com jan/2018 | a exclusão do primeiro mês |
| APS: referência com CID + CIAP | o denominador |
| CAPS: referência = capítulo F inteiro | o denominador |
| CAPS: no máximo 10 registros de jogo por CAPS-mês | picos de um único serviço lançando muita coisa de uma vez |
| CAPS: só RAAS iniciadas no mês | uma aproximação de casos novos, sem o acompanhamento |
| CAPS: F63 sem especificação somados ao desfecho, ou como comparador | reclassificação dentro de F63 |
| controle só com F63.3 | a composição do controle |

## 7. Secundário: a alta começou primeiro na APS?

Comparamos a data da quebra que abre o último segmento na APS e nos CAPS. A defasagem é CAPS menos APS, em meses.

O bootstrap é conjunto: a réplica b da APS e a réplica b dos CAPS reamostram os mesmos meses do calendário. Isso mantém os choques que atingem as duas redes ao mesmo tempo. O intervalo da defasagem sai dessas réplicas pareadas, e contamos em quantas o CAPS vem depois.

O teste é feito no modelo principal (jogo / referência). O modelo jogo vs controle entra só como sensibilidade.

Mesmo que a APS venha antes, isso mostra quando cada rede passou a registrar jogo. Não mostra pacientes indo de uma para a outra.

## 8. Secundário e exploratório: feminização

Queremos saber se a proporção de mulheres cresce mais no jogo do que na referência. O modelo é uma logística das mulheres sobre o total, com as séries de jogo e de referência empilhadas, dummies de mês e uma tendência própria para cada grupo. O erro-padrão é de Driscoll-Kraay (escores somados por mês e Newey-West). O teste é a razão de OR anual entre jogo e referência.

A janela principal, 2022–2026, foi escolhida depois de ver os dados, porque é quando o volume deixa de ser pequeno. Por isso o resultado é exploratório, e mostramos também as janelas a partir de 2021 e de 2023.

## 9. Figuras

| Arquivo | O que mostra |
|---|---|
| `fig1_volume` | registros por mês em relação à média de 2018: jogo, referência e controle |
| `fig2_taxa_joinpoint` | taxa por 10 mil registros de referência com a linha ajustada. A linha tracejada é a quebra final e a faixa azul é o intervalo bootstrap dela. Os pequenos degraus na linha vêm do indicador de pandemia. As leis aparecem só como referência de época |
| `fig3_robustez` | os cenários da seção 6 |
| `fig4_mulheres` | % de mulheres por ano, jogo e referência. Não tem intervalo de confiança porque os registros de um mesmo paciente não são independentes |

## Resultados

Rodada de 15/11/2026, com 999 réplicas. Os intervalos são do bootstrap, a não ser quando indicado.

O sistema cresceu, mas pouco perto do jogo. No último segmento, a referência sobe 19% ao ano na APS e 6% ao ano nos CAPS.

O registro de jogo cresceu bem mais que o sistema, nas duas redes:

| | Início da alta | VPA do último segmento |
|---|---|---|
| APS | jun/2022 (abr/2021 a dez/2022) | +100% (91 a 110) |
| CAPS | mai/2024 (jun/2023 a jan/2025) | +181% (107 a 367) |

O controle quase não se mexe. Na APS sobe 4% ao ano. Nos CAPS sobe 35% ao ano, mas só a partir de jul/2025. A razão jogo : controle cresce 91% ao ano na APS desde jun/2022 (84 a 99) e 151% ao ano nos CAPS desde dez/2023 (120 a 204).

Nos CAPS, os F63 sem especificação também sobem: 30% ao ano desde out/2021 (IC HAC 26 a 34). É o resultado que mais pesa contra. Com eles somados ao desfecho, a alta cai de 181% para 62% ao ano. Usados como comparador, a razão cresce 106% ao ano, contra 151% com o controle. O jogo continua crescendo mais que tudo, mas parte da alta do F63.0 provavelmente é reclassificação dentro de F63.

Nos outros cenários de robustez, o último segmento fica entre +100% e +103% na APS e entre +145% e +237% nos CAPS. A quebra cai entre jun/2022 e jul/2023 na APS e entre nov/2023 e mai/2024 nos CAPS.

A APS vem antes. Pelo modelo principal, a defasagem é de 23 meses (11 a 39), e o CAPS vem depois em 995 das 999 réplicas. No modelo jogo vs controle, a defasagem é de 18 meses (8 a 35).

A feminização aparece só na APS e é frágil. A participação das mulheres no jogo foi de 22% em 2022 para 39% em 2023 e depois ficou entre 46% e 48%. Na referência ficou em 70%. A razão de OR anual é 1,14 (1,03 a 1,27, IC Driscoll-Kraay) a partir de 2022 e 1,18 a partir de 2021. A partir de 2023 cai para 1,09 (0,99 a 1,20). Parece mais um salto em 2023 do que uma tendência. Nos CAPS não há sinal em nenhuma janela.

## Limitações

Registro não é pessoa. Sem ligar pacientes, não dá para separar caso novo de acompanhamento. A RAAS iniciada no mês é só uma aproximação.

O controle negativo pega mudanças gerais de codificação. Não pega uma mudança que favoreça o F63.0 em particular: mais atenção ao tema na mídia, a Lei 14.790, orientações do Ministério da Saúde. Daqui não dá para separar aumento de demanda de aumento de reconhecimento do problema. Por isso os resultados devem ser lidos como registro de atendimento, não como ocorrência.

Nos CAPS, os F63 sem especificação também crescem, e somá-los ao F63.0 reduz bastante a alta. Uma parte do que aparece como jogo é provavelmente codificação mais específica.

Antes de 2021, os registros de jogo nos CAPS são poucos e concentrados. Em 2020 um único CAPS responde por metade do ano. O maior lançamento num mês foi de 150 registros, em ago/2022. Por isso a quebra de 2020 no modelo jogo vs controle dos CAPS e a % de mulheres antes de 2021 não devem ser interpretadas.

Parte do crescimento nos CAPS vem de mais serviços passando a usar o código, não só de mais registros por serviço. Os dados por estabelecimento permitem separar as duas coisas, mas isso não foi feito aqui.

O modelo não permite saltos de nível, só mudanças de inclinação. Um salto, como o de jan/2025, entra como inclinação. Algumas séries têm autocorrelação nos resíduos (Ljung-Box p < 0,05), o que sugere que a linha quebrada não descreve tudo.

O desenho é ecológico. Não estima prevalência nem o efeito de uma lei.

O plano não foi registrado antes. Escolhas como o máximo de 2 quebras, a regra dos 2 pontos de BIC, os códigos do controle e a janela da feminização foram feitas durante a análise.
