# Guitar Hero IRL — Design

> v0.3 · 2026-11-11 · [requisitos](requirements.md) · [tarefas](tasks.md) · [avaliação](evaluation.md)

## Pipeline (por quadro)

webcam → mãos → braço (pose H: quadro-chave K → atual) → gestos → calibração **ou** jogo → desenho → Qt.

## Métodos

| Etapa | Método |
|---|---|
| Mãos | MediaPipe Hand Landmarker (VIDEO). Sem papel fixo por lateralidade: a posição decide. |
| Achar o braço | Só YOLOE prompt-free (classe *guitar*): máscara → PCA + perfil de largura → trecho estreito = braço; bordas por ajuste robusto; lado da cabeça = ponta com menos massa além dela. Roda numa thread, com aquecimento da GPU ao carregar (~0,1 s para achar). Sem SAM: na gravação, o YOLOE sozinho achou o braço em 100% dos quadros, com o mesmo comprimento que YOLOE + SAM; o SAM sozinho falhou em 13% e precisa da caixa do YOLOE. |
| Seguir o braço | Pontos LK presos a K + homografia K → atual (RANSAC); previsão quadro a quadro para movimento rápido/borrado; mão excluída; a detecção atrasada é "alcançada" pelos quadros guardados; perdido (violão fora do quadro ou coberto) → continua tentando reencontrar o **mesmo** K, sem limite de tempo: ORB (teste da razão) contra K a cada 3 quadros e na região que a segmentação aponta; a calibração é mantida. Se o violão está visível mas não casa com K por 2 s, vira um K novo e a calibração é transferida (ORB entre os dois K) e reajustada pelos trastes. Só "recalibrar" pede as pestanas de novo. |
| Calibração | Pestana: indicador esticado e perpendicular cruzando o eixo; dedo na casa n = traste n−1 + β·casa (β = 0,75). Duas pestanas (1 e 12) → mapa afim g ↔ s, com g = 1 − 2^(−n/12). Rejeita pestanas trocadas e a mesma posição duas vezes. |
| Ajuste pelos trastes | Retifica a faixa do braço em K; perfil de linhas finas transversais (mediana através do braço); um pico perto de cada traste previsto; refaz o ajuste **projetivo** s = (a·g + b)/(c·g + 1). Aceito só se concordar com as pestanas (meia casa). Corrige a perspectiva. Também leva a calibração para um K novo: os trastes previstos vão pela homografia entre os K e o ajuste corrige; aceito só se cada traste mexeu < 1/3 de casa (um deslocamento de uma casa inteira também "encaixa"). |
| Suavização | Filtro One Euro nos 4 cantos do braço projetados na imagem → homografia suavizada (usada no desenho e no acerto). Parado, corta a tremedeira; em movimento, quase não atrasa. Medido na gravação (min_cutoff 1 Hz, beta 0,1): tremedeira 1,74 → 0,50 px; atraso médio 2,7 px (máx 9) em movimento. |
| Cordas | 6 linhas entre as bordas, margem de 9% da largura. |
| Acerto | Ponta de dedo (exceto da mão que dedilha) na célula alvo por ≥ 2 quadros (folga 0,25 casa / 0,5 corda) **e** a corda tocada a ≤ 0,35 s. |
| Dedilhado | Mão que dedilha = a que está sobre o corpo (mediana além da "casa 13"). Vale **qualquer dedo** (polegar a mínimo) que atravesse a corda alvo **para baixo** (em direção ao chão, corda 6 → 1): estava na corda ou acima e desce ≥ 0,5 corda em 0,3 s. `game.pluck_direction: any` aceita os dois sentidos. |
| Gestos | Punho: as 4 pontas perto do pulso (‖ponta−pulso‖ < 1,3·‖MCP−pulso‖); aberta: > 1,55. 2 quadros seguidos. Punho ≤ 1,5 s e depois mão aberta = avançar; punho 3 s = recalibrar. |

## Coordenadas

imagem ↔ K (homografia) ↔ eixo (s, t) ↔ braço (g ao longo, w ∈ [0, 1] através) ↔ música (casa n̂, corda k̂).
Casa n = intervalo (n−1, n]; corda 6 = Mi grave, na borda de cima da imagem (configurável).

## Exercícios

`exercises/<nome>.json`:

```json
{"nome": "...", "descricao": "...", "notas": [{"corda": 6, "casa": 5}, {"corda": 6, "casa": 8}]}
```

corda 1 (Mi agudo) a 6 (Mi grave); casa 1 a 12.

## Código

```
GuitarHeroIRL.pyw   abre com dois cliques (usa o .venv; erros em logs/app.log)
config.yaml         configuração (padrões em ghirl/config.py; --set secao.chave=valor)
exercises/          exercícios JSON
ghirl/
  app.py       pipeline e linha de comando      ui.py        janela Qt
  capture.py   webcam, replay, gravação         hands.py     MediaPipe e gestos
  neck.py      segmentação → geometria          tracking.py  rastreador e estimador
  calib.py     trastes, pestana, ajuste         game.py      exercícios, dedilhado, jogo
  geometry.py  geometria 2D e tipos             config.py    configuração
test_ghirl.py       10 testes sintéticos (sem câmera/GPU)
```
