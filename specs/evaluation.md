# Guitar Hero IRL — Plano de Avaliação Experimental

> Versão 0.2 (revisada: só imagens, sem ArUco, violão de aço, calibração de 2 pontos) · 2026-11-11
> Requisitos: [requirements.md](requirements.md) · Design: [design.md](design.md) · Tarefas: [tasks.md](tasks.md) (Marco 5)

## 1. Objetivos e hipóteses

Pergunta central: **com modelos prontos (sem treino), uma webcam comum e só imagens, é possível manter uma grade de casas precisa e estável sobre um violão em movimento, em tempo real, a ponto de viabilizar um jogo?**

| # | Hipótese | Experimento |
|---|---|---|
| H1 | O módulo A (SAM + LK + homografia) tem erro mediano de traste ≤ 25% do espaçamento local da casa na condição base, contra a anotação manual. | E1 |
| H2 | A tem menos *jitter* e menos perda de rastreamento que B-quadro e C, que detectam o braço quadro a quadro. | E1, E2 |
| H3 | Com o violão de frente, o modelo afim de 2 pontos basta; o projetivo (3p) só reduz o erro de forma relevante com ângulo de câmera ≥ 20°. | E3 |
| H4 | A calibração por pestana é capturada com sucesso em ≥ 80% das tentativas na condição base. | E4 |
| H5 | Casas são distinguíveis pela visão com ≥ 90% de acerto; cordas exatas, não (< 80%); zonas de cordas, sim (≥ 90%). | E5 |
| H6 | O pipeline completo roda a ≥ 20 FPS, com latência de processamento ≤ 80 ms. | E6 |

**Ground truth = anotação manual** (§3). Não há marcadores no violão. B e C são **opcionais**; se não forem implementados, H2 fica restrita a A × A2 (SAM 2 vídeo, também opcional) e às ablações de A (com e sem tracking, com e sem re-segmentação).

## 2. Conjunto de dados

### 2.1 Montagem

- Câmera fixa (tripé ou topo do monitor), de frente para o usuário. Registrar modelo, resolução (1280×720), FPS, exposição, distância e altura.
- Violão: Tagima Dallas (aço). Medir uma vez com régua: escala (2 × nut → traste 12), largura no nut e no traste 12. Esses valores só entram na estimativa de inclinação e na validação da calibração.
- Nada é colado no violão.

### 2.2 Fatores e níveis

| Fator | Níveis |
|---|---|
| Ângulo do violão em relação à câmera (guinada) | **0°** (de frente, base) · 20° · 40° |
| Iluminação | **normal** (luz ambiente + luz frontal, base) · contraluz (janela atrás) · fraca (só luz ambiente à noite) |
| Movimento do violão | parado · **lento** (base; ~10 cm/s, ~10°/s) · rápido (~30 cm/s, ~30°/s, inclui sair parcialmente do quadro) |

**Matriz reduzida (um fator por vez a partir da base):** base + 2 ângulos + 2 iluminações + 2 movimentos = **7 condições**. Cada condição tem **3 repetições** de **60 s**, com o roteiro:
1. 0–10 s: violão parado (jitter).
2. 10–20 s: calibração (pestana na casa 1 e depois na 12; no modo 3p, também na 5).
3. 20–50 s: tocando o chart da pentatônica (60 BPM), com o movimento da condição.
4. 50–60 s: tirar o violão do quadro por ~3 s e recolocar (perda e recuperação).

Total: 21 clipes, ~21 min de vídeo. A matriz fatorial completa (27 condições) fica como extensão se houver tempo.

Clipes adicionais:
- **E4** (pestana): 20 tentativas de calibração na base + 10 em cada condição de iluminação.
- **E5** (viabilidade de cordas): ver §5.5.
- **E6** (latência fim-a-fim): ver §6.

### 2.3 Armazenamento

`data/eval/<condicao>/<rep>/` com `video.mkv`, `timestamps.json`, `camera.json`, `annotations.json` e `notes.md`. O inventário fica em `data/eval/index.csv`. Os vídeos ficam fora do controle de versão.

## 3. Ground truth por anotação manual

- **Amostragem**: 1 quadro por segundo nos trechos parados e de calibração; 2 quadros por segundo nos trechos em movimento. Por clipe, ≈ 90 quadros. No conjunto, ≈ 1.900 quadros.
- **O que anotar em cada quadro** (ferramenta da T5.1): interseções do **nut** e dos **trastes 1, 5, 7 e 12** com as duas bordas do braço (10 cliques), mais o papel das mãos.
- **Trastes não anotados** (2, 3, 4, 6, 8–11): interpolados pelo modelo 12-TET com uma **homografia 1D ajustada aos 5 trastes anotados** em cada borda. Isso também dá um GT com perspectiva. O resíduo do ajuste nos 5 trastes anotados mede a consistência da anotação e entra no relatório.
- **Intervalos de visibilidade**: marcar quando o violão está visível / parcialmente fora / fora do quadro (para a métrica de perda).
- **Células tocadas** (para E5 e para a taxa de acerto do jogo): em cada nota do chart, anotar a célula que o dedo realmente ocupou e se ele estava lá no tempo da nota.
- **Variabilidade do anotador**: 10% dos quadros anotados duas vezes, em dias diferentes. Reportamos o desvio mediano em px, que serve de piso de erro: um método não pode ser avaliado com precisão melhor que isso.
- Limitação: o GT é esparso no tempo (1–2 Hz). As métricas de erro usam só os quadros anotados. O jitter é medido sem GT (§4.2).

## 4. Métricas

### 4.1 Erro de posição das casas
Para cada quadro anotado e cada traste n ∈ {1..12}: e_n = distância entre o ponto médio previsto e o do GT (px). Também em forma normalizada: **ê_n = e_n / Δ_n**, onde Δ_n é o espaçamento local da casa no GT, em px. A versão normalizada é a métrica principal, porque compara condições com escalas diferentes. Reportamos mediana, P90 e média por traste, por condição e por método; separamos os trastes anotados diretamente (1, 5, 7, 12) dos interpolados. Componentes: erro **ao longo** do eixo (deslizamento) e **transversal**, separados.

### 4.2 Estabilidade (jitter)
Sem GT, nos trechos parados: desvio-padrão temporal da posição prevista de cada traste (px), com a média sobre os trastes, em todos os quadros (não só os anotados). Em movimento: desvio-padrão da **segunda diferença** temporal das posições previstas (aceleração aparente), que separa o tremor do movimento suave real.

### 4.3 Perda de rastreamento
- **Taxa de perda**: % de quadros em PERDIDO (ou sem pose) dentro dos intervalos anotados como "violão visível".
- **Eventos de perda por minuto** e **tempo médio de recuperação** (do fim do intervalo "fora do quadro" até RASTREANDO).
- **Recuperações sem recalibrar**: % dos eventos de perda recuperados antes de `T_relock`.
- **Falsos "ok"**: quadros anotados com status ok e ê mediano > 0,5 (o sistema "acha" que está certo).

### 4.4 Calibração por pestana
- **Taxa de sucesso**: % das tentativas que terminam em ≤ 30 s **e** com ê mediano ≤ 0,25 no primeiro quadro anotado depois da calibração.
- **Tempo até a captura** de cada pestana.
- **Falsos positivos**: capturas feitas sem pestana real (verificadas no vídeo).
- **Erro da medida**: |s_i − s_GT(g_dedo(n))| em px e em fração da casa, que também permite estimar o β real.

### 4.5 Acerto das notas no jogo
Contra a anotação das células realmente tocadas (§3):
- **Precisão/recall** do julgamento "acertou × errou".
- **Matriz de confusão de casas** (n̂ arredondado × casa real) e **de cordas** (k̂ × corda real).
- **Taxa de acerto** por modo de cordas (zonas, exata).
- Distribuição do instante de chegada do dedo em relação ao tempo da nota, para calibrar as janelas.

### 4.6 Desempenho
- **FPS** do pipeline (média, P5) ao vivo e em replay "tempo real".
- **Latência por estágio** (P50/P95 em ms): captura→disponível, MediaPipe, tracker, relocalização, SegWorker (por chamada e por modelo), geometria, jogo, render.
- **Latência de processamento**: `ts` do quadro → `imshow` concluído.
- **Latência fim-a-fim** (§6).
- Uso de VRAM e CPU (amostrado).

## 5. Experimentos

### 5.1 E1 — Precisão e estabilidade da localização (H1, H2)
- Métodos: **A** (SAM + LK), **A2** (SAM 2 vídeo, opcional), **B-quadro** e **B+track** (opcionais), **C** (opcional).
- Para isolar a localização da calibração, cada método roda em dois regimes: (i) **calibração oráculo** (nut e traste 12 tirados da anotação do quadro de calibração) e (ii) **calibração real** (pestana do próprio clipe).
- Saídas: tabela método × condição com ê mediano/P90 e jitter; boxplots de ê por condição; erro por traste (curva ê_n × n).
- Teste estatístico: Wilcoxon pareado por clipe (A × cada alternativa), com correção de Holm.

### 5.2 E2 — Robustez de rastreamento (H2)
- Mesmos métodos e clipes de E1, foco no movimento (parado, lento, rápido) e nos trechos de saída/retorno.
- Saídas: taxa de perda, eventos/min, tempo de recuperação, % de recuperação sem recalibrar, falsos "ok".
- Ablações de A: sem máscara das mãos; sem re-segmentação; período de re-segmentação 1/2/5 s; com e sem filtro One Euro (jitter × atraso).

### 5.3 E3 — Modelo de calibração e perspectiva (H3)
- Modelos: **2p afim (padrão)**, 2p + afunilamento, 3p projetivo. Nos clipes de E3, a calibração é gravada com as três pestanas (1, 12, 5); o 2p ignora a terceira.
- Fator principal: ângulo 0°/20°/40°.
- Parte **sintética** (da T3.2): erro × (θ, ruído) para os três modelos, comparando teoria e simulação.
- Também: estimativa de β a partir das medidas contra o GT; sensibilidade do erro final a β ∈ {0,5; 0,6; 0,7; 0,75; 0,8; 0,9}.
- Decisão de saída: se o 2p atender a RNF-04 na base e a 20°, o padrão continua 2p.

### 5.4 E4 — Detecção da pestana (H4)
- 20 tentativas na base e 10 em cada iluminação; registrar o motivo da rejeição por quadro (indicador dobrado, ângulo, cobertura, instabilidade, mão não detectada).
- Saídas: taxa de sucesso, tempo até a captura, falsos positivos, histograma de motivos de rejeição, taxa de detecção do MediaPipe durante a pestana.
- Comparação com a calibração manual por clique (erro e tempo).

### 5.5 E5 — Viabilidade de distinguir cordas e casas (H5) · alimenta a T4.5
- Protocolo: o usuário coloca **um dedo** em células indicadas na tela (amostra de 24 células cobrindo cordas 1–6 × casas {1, 3, 5, 7, 9, 12}, mais as 12 células da pentatônica) e segura 2 s, a **0,7 m** e a **1,0 m**, na base. Duas repetições. A célula indicada é o GT (sem anotação extra).
- Saídas: distribuição de n̂ − n e k̂ − k; matrizes de confusão; acurácia para os modos zonas e exata em função de ε_n, ε_k e γ (curvas); escolha dos valores padrão.
- Jogo completo: taxa de acerto das notas (§4.5) nos trechos de chart dos clipes da matriz.

### 5.6 E6 — Desempenho (H6)
- Ao vivo: 3 execuções de 60 s em RASTREANDO + jogo ativo, para cada configuração relevante (A com SAM 3, A com SAM 2.1, B, C).
- Replay "o mais rápido possível": throughput máximo de cada estágio.
- Saídas: tabela de FPS e latência por estágio (P50/P95); gráfico de barras empilhadas do orçamento por quadro.

## 6. Latência fim-a-fim (*glass-to-glass*)

1. Um cronômetro em milissegundos (num celular) aparece no campo da câmera e, ao mesmo tempo, na foto junto com a tela do app.
2. Uma segunda câmera (celular em câmera lenta) ou uma foto capta simultaneamente o cronômetro real e o cronômetro exibido pelo app.
3. Latência = diferença entre os dois valores. 20 amostras; reportar mediana e P90.
4. Com isso calibramos `camera.latency_ms` (latência da câmera = fim-a-fim − latência de processamento − latência de exibição estimada).

## 7. Apresentação no relatório

- Tabela 1: condições e clipes (inventário).
- Tabela 2: E1, ê mediano/P90 e jitter por método × condição.
- Tabela 3: E2, perda e recuperação por método × movimento.
- Tabela 4: E3, erro por modelo de calibração × ângulo (+ curva sintética).
- Tabela 5: E4, taxa de sucesso da pestana e motivos de rejeição.
- Tabela 6: E5, acurácia de casas/cordas por distância e modo + matriz de confusão.
- Tabela 7: E6, FPS e latência por estágio.
- Figuras: quadros de exemplo com o overlay previsto × anotação em cada condição; boxplots; curva de erro por traste; orçamento por quadro.

## 8. Ameaças à validade

- **Um único usuário, violão e câmera**: a generalização é limitada; deixar explícito.
- **GT manual esparso e ruidoso**: erro do anotador da ordem de alguns px. Mitigação: dupla anotação de 10%, piso de erro reportado, ajuste 12-TET para checar a consistência.
- **Viés do anotador** (a mesma pessoa projeta e anota): anotar sem ver o overlay do método (a ferramenta mostra só o vídeo bruto).
- **Variação da webcam** (exposição automática que o driver não deixa desligar): registrar as propriedades efetivas por clipe.
- **Efeito de aprendizado** do usuário entre as repetições do jogo: ordem das condições aleatorizada.

## 9. Escopo mínimo se o tempo apertar

Prioridade decrescente:
1. A na condição base e em movimento (E1 e E2 reduzidos), com anotação só dos trastes 1 e 12.
2. E6 (desempenho).
3. E4 (pestana) e E3 (2p × 3p).
4. E5 (cordas/casas).
5. Ângulo e iluminação completos.
6. **B e C (opcionais)** e A2.
