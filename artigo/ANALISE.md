# Jogo patológico na APS e nos CAPS, Brasil, 2018–2026: plano de análise

Três arquivos:

- `analise.py` roda tudo.
- este documento explica o quê e o porquê.
- `resultados.txt` é a saída. As figuras ficam em `figuras/`.

As seções numeradas abaixo são as mesmas do script e do `resultados.txt`.

```
uv run python artigo/analise.py          # ~15 min; --B 99 para um teste rápido
```

## 1. Dados

Só usamos dados públicos e agregados, sem identificação. Não há ligação de pacientes, por isso não é preciso submeter ao CEP (Res. CNS 510/2016, art. 1º, III e V).

| Série | Fonte | Unidade | Período |
|---|---|---|---|
| APS | SISAB, relatório público de produção (atendimento individual por CID) | atendimentos/mês, Brasil | jan/2018–set/2026 (105 meses) |
| CAPS | SIA-PS, RAAS (arquivos .dbc do FTP do DATASUS) | registros de RAAS/mês por CID principal, Brasil | jan/2018–jun/2026 (102 meses) |

- **RAAS.** O script lê apenas os bytes de 5 colunas do .dbf: CNES, data do atendimento, sexo, CID principal e data de início. O cartão SUS (CNS_PAC) e a data de nascimento nunca são lidos.
- **Meses excluídos.** Jul–ago/2026 ficam de fora porque ainda estão incompletos (a RAAS é enviada com atraso). Falta também 1 arquivo dos 2.808 (AP, dez/2023).

**Grupos (CID-10)**

| Grupo | Códigos | Papel |
|---|---|---|
| Jogo | F63.0 | desfecho |
| Referência | F32, F33, F41, F43, F10, F20 | "o sistema": demanda geral de saúde mental, usada como denominador |
| Controle negativo | F63.1–F63.3 (piromania, cleptomania, tricotilomania) | mesmo agrupamento (F63), mesmas regras de codificação, sem relação esperada com apostas |

**Unidade de análise.** A unidade é o **registro**, não a pessoa. Um paciente em seguimento gera vários registros.

## 2. Modelo comum: regressão segmentada (joinpoint)

É o mesmo modelo do Joinpoint do NCI, comum na CSP, aqui para contagens mensais:

```
log E[y_t] = log(denominador_t) + sazonalidade (11 dummies de mês) + pandemia (mar/2020–dez/2021)
             + β·t + Σ_k δ_k·(t − τ_k)+
```

- **Família.** Binomial negativa (NB2) para contagens com denominador (offset). Binomial com logit quando o desfecho é uma fração (jogo / (jogo + controle)).
- **Quebras.** As posições τ são escolhidas por busca exaustiva, com pelo menos 12 meses entre quebras e das pontas. O número de quebras K vai de 0 a 2 e é escolhido por BIC (QBIC no binomial). Fica o menor K a menos de 2 pontos do mínimo.
- **Medida.** VPA, a variação percentual anual de cada segmento: 100·(e^(12·inclinação) − 1). VPAM é a média ponderada no período todo (equivale à AAPC).
- **Inferência.**
  - Erros-padrão HAC (Newey-West, 6 defasagens) para a autocorrelação, condicionais às quebras.
  - Bootstrap por blocos móveis dos resíduos (blocos de 6 meses, 999 réplicas), refazendo a busca das quebras em cada réplica. Assim o IC da VPA e o IC da data da quebra incorporam a incerteza sobre onde a quebra está.
- **Diagnóstico.** Ljung-Box (12 defasagens) nos resíduos de Pearson.

## 3. Descritiva

Tabela anual com contagens, taxa de F63.0 e do controle por 10 mil registros de referência, taxa por milhão de habitantes (população IBGE) e % de mulheres.

## 4. H1 (efeito principal): o registro de jogo cresceu além do crescimento do sistema?

**Hipótese.** A taxa de F63.0 por registro de referência passa a crescer a partir de algum momento, isto é, há uma quebra com aumento de inclinação, e o segmento final tem VPA > 0.

**4a. Crescimento do sistema.** O mesmo modelo, aplicado às contagens do denominador sem offset, mede quanto o próprio sistema cresceu. Usamos:
- a referência;
- a referência ampliada com CIAP (APS);
- todas as RAAS (CAPS).

**4b. Modelo principal.** Contagem de F63.0 com offset log(referência), NB. Com o offset, o crescimento do sistema já sai descontado: a VPA é a do jogo *por unidade de demanda de saúde mental*.

| Teste | Estatística | Rejeita H0 se |
|---|---|---|
| Existe mudança de tendência? | BIC de K = 0 contra K ≥ 1 | K escolhido ≥ 1 |
| A mudança é significativa? | Wald (HAC) de δ = 0 na quebra | p < 0,05 |
| O jogo cresce no segmento final? | Wald (HAC) da inclinação final = 0; IC bootstrap da VPA | p < 0,05 e IC bootstrap acima de 0 |

## 5. Especificidade: controle negativo

Uma alta da taxa de jogo poderia ser só artefato de codificação: adoção do CID no e-SUS, mudança no preenchimento da RAAS, capacitação da rede. Um artefato desses atingiria também os outros códigos F63.

- **5a.** O controle (F63.1–F63.3) passa pelo mesmo modelo do H1. Esperado: sem alta comparável.
- **5b. Teste direto.** Modelo binomial da fração F63.0 / (F63.0 + F63.1–F63.3), com logit. A inclinação é exatamente a diferença entre as taxas de crescimento do jogo e do controle. A VPA é o crescimento anual da razão jogo : controle. Rejeita-se "crescimento igual ao do controle" se a VPA do segmento final for > 0 (Wald HAC e IC bootstrap).

## 6. Robustez

Refazemos a VPA do segmento final, com a data da quebra, nos cenários abaixo. A conclusão se mantém se a VPA continuar alta e a quebra final ficar na mesma época.

| Cenário | Para quê |
|---|---|
| até 3 quebras / 1 quebra fixa | dependência da escolha de K |
| sem os meses da pandemia | distorção de 2020–2021 |
| série até dez/2025 | meses recentes ainda sendo consolidados |
| HAC com 12 defasagens | autocorrelação mais longa |
| APS: referência ampliada (CID + CIAP) | escolha do denominador |
| CAPS: referência = capítulo F inteiro | escolha do denominador |
| CAPS: no máx. 10 registros de F63.0 por CAPS-mês | picos por lançamento em lote de um único CAPS |
| CAPS: só RAAS abertas no mês | proxy de casos novos, sem seguimento |
| controle = só F63.3 | composição do controle |

## 7. Secundário: a alta começou na APS?

**Hipótese.** A quebra que inicia o segmento final ocorre antes na APS do que no CAPS (porta de entrada → serviço especializado).

**Medida.** Defasagem CAPS − APS, em meses, entre as datas das quebras finais.

**Teste.**
- IC95% da defasagem pelos bootstraps das duas séries (réplicas independentes).
- Proporção de réplicas em que o CAPS vem depois.

**Modelos.** Fazemos o teste nos dois pares de modelos: H1 (jogo/referência) e 5b (jogo vs controle).

## 8. Secundário: feminização

**Hipótese.** A proporção de mulheres entre os registros de jogo cresce mais do que entre os registros de referência.

**Modelo.** Regressão logística das mulheres sobre o total, com as séries de jogo e de referência empilhadas. Inclui dummies de mês e uma tendência anual própria para cada grupo. O período é 2022–2026, quando o volume de F63.0 deixa de ser pequeno.

**Inferência.** EP de Driscoll-Kraay: escores somados por mês, Newey-West com 6 defasagens.

**Teste.** Razão de OR anual jogo/referência = 1 (Wald).

## 9. Figuras

| Figura | Conteúdo |
|---|---|
| `fig1_volume` | registros mensais relativos a 2018: jogo × referência (sistema) × controle |
| `fig2_taxa_joinpoint` | taxa por 10 mil registros de referência, com o ajuste segmentado, a quebra final e o IC bootstrap da quebra; marcos legais só como referência |
| `fig3_robustez` | floresta da seção 6 |
| `fig4_mulheres` | % de mulheres por ano, jogo × referência (IC de Wilson) |

## Principais resultados (execução de 15/11/2026, 999 réplicas; detalhes em `resultados.txt`)

**Sistema.** A referência cresceu +19% ao ano na APS e +6% ao ano nos CAPS, no segmento final.

**H1.** A taxa de F63.0 por registro de referência acelera:

| Série | Início da aceleração (IC95% bootstrap) | VPA do segmento final, IC95% HAC | IC95% bootstrap | Mudança de inclinação |
|---|---|---|---|---|
| APS | jun/2022 (mai/2021–out/2025) | +101% (90; 112) | (64; 136) | p < 0,001 |
| CAPS | mai/2024 (jul/2023–fev/2025) | +181% (155; 209) | (103; 362) | p < 0,001 |

**Especificidade.** O controle praticamente não muda: +5% ao ano na APS; nos CAPS, +35% ao ano só a partir de jul/2025. A razão jogo : controle cresce:
- APS: +91% ao ano desde out/2021 (bootstrap 84; 100);
- CAPS: +151% ao ano desde dez/2023 (bootstrap 118; 203).

**Robustez.** Em todos os cenários a VPA final fica entre +100% e +110% na APS e entre +145% e +237% nos CAPS, com a mesma época de quebra.

**Onde começou.**
- Razão jogo : controle: a APS vem 26 meses antes (IC 12–35), e o CAPS vem depois em 100% das réplicas.
- Com a referência como denominador: 23 meses, mas o intervalo é impreciso (IC −18 a 37; CAPS depois em 84% das réplicas).

**Feminização.**
- APS: a proporção de mulheres no jogo vai de 22% para 48%, contra 70% estáveis na referência; razão de OR 1,14 ao ano (1,03–1,27), p = 0,013.
- CAPS: sem evidência (0,92; 0,78–1,09).

## Limitações

- **Registros não são pessoas.** Sem ligação de pacientes, não dá para separar casos novos de seguimento. RAAS aberta no mês é só um proxy.
- **O que o controle negativo neutraliza.** Neutraliza mudanças gerais de codificação. Não neutraliza uma campanha de codificação específica para jogo.
- **Desenho.** Ecológico e descritivo do uso de serviços. Não estima prevalência nem o efeito causal de uma lei. Os marcos legais aparecem só como contexto.
- **Lançamentos em lote.** No CAPS, a série de F63.0 tem picos de lançamentos em lote (um único CAPS chega a 99 registros num mês). Isso explica quebras extras em 2019–2020 quando se permite K maior. O segmento final não muda (ver robustez).
- **Autocorrelação residual.** Aparece em parte das séries. É tratada com HAC e com bootstrap por blocos.
