# Guitar Hero IRL — Avaliação

> v0.3 · 2026-11-11 · [requisitos](requirements.md) · [design](design.md) · [tarefas](tasks.md)

## Dados

3–5 sessões gravadas (`python -m ghirl --record data/rec/<nome>`), 30–60 s cada: violão parado, em
movimento, inclinado, com a mão sobre o braço, e um exercício tocado com alguns erros de propósito.
Ground truth por anotação manual (sem marcadores): clicar nut e trastes 1, 5, 7 e 12 nas duas bordas
do braço em ~20 quadros.

## Métricas

| O quê | Métrica |
|---|---|
| Braço | % de quadros rastreados; tempo para achar; recuperações após perda |
| Calibração | erro (px e % da casa) dos trastes previstos × anotados: só pestanas × com ajuste pelos trastes |
| Jogo | acertos corretos, acertos falsos (nota errada ou corda não tocada) e notas perdidas |
| Desempenho | FPS e ms por etapa (mãos, braço, resto), mostrados no painel |

## Resultados até agora

- Braço achado em ~0,2 s; 0 quadros perdidos na gravação de 30 s.
- Violão fora do quadro por 8 s (simulado na gravação): reencontrado 0,2 s após voltar, com o mesmo quadro-chave (calibração mantida).
- Ajuste pelos trastes no vídeo real, com pestanas simuladas (ruído de 4 px): erro máximo dos trastes
  ~18 px → ~6 px.
- Calibração levada para outro quadro-chave (quadros 100 → 600 da gravação) e reajustada pelos trastes:
  13 trastes, 0,1 px da referência.
- Suavização One Euro na gravação: tremedeira com o violão parado 1,74 → 0,50 px; atraso médio de 2,7 px
  (máx 9 px) com o violão em movimento.
- ~28–35 ms por quadro com mãos + braço (≈ 30 FPS).
