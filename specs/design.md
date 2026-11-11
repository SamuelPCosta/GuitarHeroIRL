# Guitar Hero IRL — Design

> Versão 0.2 (revisada: só imagens, sem ArUco, violão de aço, calibração de 2 pontos) · 2026-11-11
> Requisitos: [requirements.md](requirements.md) · Tarefas: [tasks.md](tasks.md) · Avaliação: [evaluation.md](evaluation.md)

Os números de desempenho deste documento são **estimativas a validar** nas tasks indicadas. Trechos de código são só **contratos de interface** (assinaturas), não implementação.

---

## 1. Visão da arquitetura

```
┌──────────────┐   quadro + ts    ┌───────────────────────── Laço principal (thread main) ──────────────────────────┐
│ CameraSource │ ───────────────► │ 1 Mãos (MediaPipe) ─► 2 Braço (NeckEstimator) ─► 3 Papéis das mãos              │
│ (thread)     │  "mais recente"  │        │                     ▲   │                                              │
│ ou Replay    │                  │        │   máscara das mãos  │   │ pedidos/respostas assíncronos                │
└──────────────┘                  │        └─────────────────────┘   ▼                                              │
                                  │                         ┌──────────────────┐                                    │
                                  │                         │ SegWorker (thread│  SAM 3 / SAM 2 / YOLOE (GPU)       │
                                  │                         │ GPU, assíncrono) │                                    │
                                  │                         └──────────────────┘                                    │
                                  │ 4 Máquina de estados (calibração) ─► 5 Geometria (casas/cordas no ref. braço)  │
                                  │ 6 Jogo (scheduler, juiz, placar) ─► 7 Render (overlay + HUD) ─► 8 Log/Profiler │
                                  └─────────────────────────────────────────────────────────────────────────────────┘
```

Princípios:
1. **Laço principal nunca bloqueia** em modelos pesados (RNF-06). SAM/YOLO rodam num *worker* com fila de tamanho 1 (pedido mais recente vence).
2. **Tudo no referencial do braço** (R2). As coordenadas de imagem só aparecem na entrada (detecções) e na saída (desenho).
3. **Módulos por interface** (RNF-07): localizador, rastreador, detector de pestana, juiz e fonte de vídeo são trocáveis por configuração.
4. **Gravar e reproduzir** (RF-03): qualquer bug ou experimento roda de novo de forma determinística a partir de uma gravação.
5. **Só imagens**: nesta versão não há entrada nem saída de áudio. O metrônomo é visual.

## 2. Stack e ambiente

| Componente | Escolha | Observação |
|---|---|---|
| Python | 3.12 (venv) | SAM 3 exige ≥ 3.12; MediaPipe tem wheels cp312 para Windows. Verificar na T1.1. |
| Vídeo | OpenCV (`cv2`) com backend DirectShow (`CAP_DSHOW`) ou MSMF | DSHOW costuma expor exposição manual e MJPG. Testar ambos (T1.3). |
| Mãos | MediaPipe Tasks `HandLandmarker`, `running_mode=VIDEO` | Só CPU no Windows (Python). ~10–25 ms/quadro (estimativa). |
| Segmentação | SAM 3 (texto) — preferido; SAM 2.1 (tiny/small, ponto/caixa) — alternativa | Pelo repositório oficial ou pelo Ultralytics; decidir na T2.2 conforme instalação no Windows e latência. |
| Open-vocabulary | Ultralytics YOLOE-seg / YOLO-World (prompt "guitar") | Módulo B e gerador de prompt de caixa para SAM 2. |
| GPU | PyTorch + CUDA 12.x | RTX 3060 12 GB. |
| Testes | `pytest`, `ruff` | |
| Config | YAML (`pyyaml`) + dataclasses | |

## 3. Estrutura de pacotes (planejada)

```
ghirl/
  app/        main.py · app_state.py · config.py
  capture/    camera.py · replay.py · recorder.py
  hands/      mp_tracker.py · roles.py · filters.py (One Euro)
  neck/       base.py (interfaces) · mask_geometry.py · seg_worker.py
              sam_locator.py (A) · yolo_locator.py (B) · classic_locator.py (C)
              feature_tracker.py · relocalizer.py · estimator.py · pose_quality.py
  calib/      fret_model.py · solver.py · barre_detector.py · string_model.py · flow.py
  game/       chart.py · clock.py · scheduler.py · judge.py · score.py · flow.py
  render/     overlay.py · hud.py · debug.py · screens.py
  eval/       runner.py · annotate.py · metrics.py · plots.py
  utils/      geometry.py · profiler.py · jsonl.py
configs/default.yaml
charts/*.json
tests/
data/         (gravações, fora do controle de versão)
```

## 4. Módulos e interfaces

### 4.1 Tipos de dados centrais

```python
@dataclass
class Frame:
    image: np.ndarray        # BGR, NÃO espelhado
    ts: float                # perf_counter() no momento em que o quadro foi lido
    index: int

@dataclass
class Hand:
    landmarks_px: np.ndarray # (21, 2) pixels da imagem
    landmarks_z: np.ndarray  # (21,) profundidade relativa do MediaPipe
    handedness_raw: str      # rótulo do MediaPipe ("Left"/"Right"), sem correção
    score: float
    role: Literal["notas", "ritmo", "desconhecido"]

@dataclass
class NeckDetection:         # saída de um localizador (num quadro específico)
    frame_index: int
    mask: np.ndarray | None  # máscara binária do braço (se houver)
    axis_point: np.ndarray   # (2,) ponto no eixo
    axis_dir: np.ndarray     # (2,) unitário
    edge_a: tuple            # reta da borda (ponto, direção)
    edge_b: tuple
    extent: tuple[float, float]  # intervalo [s_min, s_max] do braço ao longo do eixo
    nut_side_hint: int | None    # +1/−1: lado da cabeça, se inferido pelo perfil de largura
    confidence: float
    latency_ms: float

@dataclass
class NeckPose:              # saída do estimador (por quadro)
    H: np.ndarray | None     # 3x3, quadro-chave K → quadro atual
    quality: float           # 0..1
    inliers: int
    reproj_rmse: float
    status: Literal["ok", "degradado", "perdido"]
```

### 4.2 Interfaces trocáveis

```python
class NeckLocator(Protocol):            # A, B, C
    name: str
    is_heavy: bool                      # True → roda no SegWorker
    def detect(self, image, hint: NeckHint | None) -> NeckDetection | None: ...

class NeckTracker(Protocol):            # LK+homografia, SAM 2 vídeo, "detecção por quadro"
    def init(self, frame: Frame, keyframe: Keyframe) -> None: ...
    def update(self, frame: Frame, exclude_mask: np.ndarray) -> NeckPose: ...

class BarreDetector(Protocol):
    def observe(self, hand: Hand, pose: NeckPose, frame_ts: float) -> BarreObservation | None: ...

class HitJudge(Protocol):
    def update(self, t_game: float, contacts: list[CellContact]) -> list[Judgement]: ...

class FrameSource(Protocol):            # CameraSource, ReplaySource
    def read_latest(self) -> Frame | None: ...
```

`NeckEstimator` compõe um `NeckLocator` e um `NeckTracker` e aplica a política de re-segmentação, detecção de perda e relocalização (§7). O resto do sistema só enxerga `NeckEstimator.update(frame, hand_mask) -> NeckPose`.

### 4.3 Responsabilidades

| Módulo | Entrada | Saída | Requisitos |
|---|---|---|---|
| `capture.camera` | config | `Frame` mais recente, FPS real | RF-01, RF-02 |
| `capture.replay/recorder` | arquivo | `Frame` com ts gravados | RF-03 |
| `hands.mp_tracker` | `Frame` | `list[Hand]`, máscara das mãos | RF-10, RF-13 |
| `hands.roles` | mãos + pose do braço | papéis | RF-11 |
| `neck.*` | `Frame`, máscara das mãos | `NeckPose`, `Keyframe` | RF-20–29 |
| `calib.*` | mãos, pose, eventos de UI | `Calibration` (u↔s, cordas) | RF-30–39 |
| `game.*` | chart, relógio, contatos | julgamentos, placar | RF-50–59 |
| `render.*` | tudo acima | imagem final | RF-04, RF-40–43 |
| `eval.*` | gravações, logs | métricas | RF-70–73 |

## 5. Sistemas de coordenadas

```
   Imagem atual (x, y) px  ◄──── H_t ────  Quadro-chave K (X, Y) px
                                              │  afim fixo (eixo do braço em K)
                                              ▼
                                       Braço em K: (s, t) px
                                       s ao longo do eixo, t perpendicular
                                              │  calibração (1D projetiva) + modelo das cordas
                                              ▼
                                       Físico: (g, w)
                                       g = u/L ∈ [0, 0.5]  (0 = nut, 0.5 = traste 12)
                                       w ∈ [0, 1] (0 = borda dos bordões, 1 = borda das primas)
                                              │
                                              ▼
                                       Contínuo musical: (n̂, k̂)
                                       n̂ = −12·log2(1 − g)   (casa contínua; traste n em n̂ = n)
                                       k̂ = corda contínua (6 … 1)
```

- **Referencial do braço** = (s, t) em K, e daí (g, w). O quadro-chave K é um retrato fixo do braço; como o espelho do braço é (quase) plano, a relação K → quadro atual é uma **homografia** H_t, válida sob qualquer movimento rígido do violão (translação, rotação no plano e fora do plano), desde que o plano continue visível.
- A calibração e o modelo das cordas são guardados **em K** e nunca mudam enquanto o rastreamento se mantém. Mover o violão só altera H_t.
- Ponto da imagem → braço: `p_K = H_t⁻¹ · p_img`, depois (s, t), depois (g, w), depois (n̂, k̂). Braço → imagem: o caminho inverso. É assim que posicionamos as bolinhas e testamos os acertos.
- Orientação: depois da calibração, +s aponta do nut para o corpo. Para destro com câmera não espelhada, a cabeça do violão aparece à **direita** da imagem e os bordões (corda 6) na borda **superior**. Os dois sentidos são inferidos (calibração e perfil de largura), com padrão configurável.

## 6. Fluxo de dados por quadro

| # | Etapa | Tempo estimado (a medir, T1.6) |
|---|---|---|
| 0 | `read_latest()` — o quadro já foi capturado pela thread | ~0 ms |
| 1 | MediaPipe HandLandmarker (VIDEO, 2 mãos) | 10–25 ms (CPU) |
| 2 | Máscara das mãos (casco convexo dilatado, em resolução reduzida) | < 1 ms |
| 3 | `NeckEstimator.update`: LK piramidal (~150 pontos) + verificação ida-e-volta + `findHomography` RANSAC; consumir resposta do SegWorker se houver; agendar novo pedido conforme a política | 3–8 ms |
| 4 | Papéis das mãos (geometria em relação ao braço) | < 1 ms |
| 5 | Máquina de estados: calibração / perda / relocalização (ORB só em PERDIDO, a cada *k* quadros) | < 1 ms (ORB: 5–15 ms) |
| 6 | Geometria: trastes 0–12, 6 cordas, centros das células → imagem via H_t; filtro One Euro | < 1 ms |
| 7 | Jogo: relógio, notas visíveis, contatos (ponta → (n̂, k̂)), juiz, placar | < 1 ms |
| 8 | Render: overlay + HUD (+ espelhamento de exibição) + `imshow` | 3–6 ms |
| 9 | Log JSONL + profiler | < 1 ms |
| | **Total** | **~20–40 ms → 25–50 FPS; a webcam limita a 30** |

O SegWorker roda em paralelo: SAM 3 ~150–500 ms por imagem e SAM 2.1-tiny ~30–60 ms na RTX 3060 (estimativas, T2.2). Por isso ele roda só em quadros-chave.

Latência: o carimbo `ts` é o momento da leitura, mas a exposição ocorreu antes (latência da câmera, tipicamente 30–100 ms). Para o jogo, usamos `t_evento = ts − latencia_camera` (configurável, medida na T5.10).

## 7. Localização e rastreamento do braço

### 7.1 Módulo A — SAM + tracking (principal)

**Inicialização (quadro-chave K):**
1. O SegWorker recebe o quadro. Com SAM 3, prompts de texto em ordem: `"guitar neck"`, `"fretboard"`, `"guitar"`. Fica a máscara com melhor *score de forma* (alongamento = razão dos autovalores da PCA; área plausível). Sem SAM 3: SAM 2.1 com caixa do YOLOE/YOLO-World ("guitar") ou com clique do usuário no braço.
2. `mask_geometry`:
   - Se a máscara cobre o violão inteiro: eixo principal por PCA, **perfil de largura** W(s) (largura da máscara em cada fatia perpendicular). O braço é o maior trecho contínuo com W(s) < τ·W_ref (τ ≈ 1,6; W_ref = mediana das larguras no quartil inferior). O lado em que, depois do trecho estreito, há um aumento curto e moderado de largura é a **cabeça** (dica `nut_side_hint`); o lado com aumento grande é o **corpo**.
   - Pontos do contorno do braço divididos em duas bordas pelo sinal de t; `cv2.fitLine` (robusto, `DIST_HUBER`) em cada borda. O eixo é a bissetriz das duas retas; a largura é a distância entre as bordas, que varia ao longo de s (afunilamento + perspectiva).
   - `minAreaRect` só como checagem/alternativa rápida.
3. Guardamos o `Keyframe`: imagem cinza, máscara do braço + cabeça (+ faixa do corpo junto ao braço), retas das bordas e eixo em coordenadas K, descritores ORB (para relocalização).

**Rastreamento por quadro (`feature_tracker`):**
- Pontos Shi-Tomasi dentro da região rastreável (máscara de K projetada por H_t, menos a máscara das mãos dilatada). Cada ponto guarda sua coordenada em K.
- LK piramidal (janela 21×21, 3 níveis) do quadro anterior para o atual, com verificação ida-e-volta (erro < 1 px).
- `cv2.findHomography(pts_K, pts_atuais, RANSAC, 3 px)` → H_t (direto de K, não compondo quadro a quadro, para limitar a deriva).
- Reposição: quando os pontos ficam abaixo de N_min (60), detectamos novos pontos no quadro atual e levamos para K com H_t⁻¹.
- A cabeça do violão (tarraxas) e a junção braço-corpo têm textura forte e raramente ficam ocultas pela mão de notas, por isso são essenciais. A cabeça do violão tem alguns graus de inclinação em relação ao espelho; o RANSAC absorve isso, e em caso de dúvida damos peso maior aos pontos do espelho.

**Qualidade (`pose_quality`)** → `quality ∈ [0,1]` combinando: inliers ≥ 25, razão de inliers ≥ 0,5, RMSE ≤ 2 px, plausibilidade de H (det > 0; escala relativa a K em [0,4; 2,5]; termos projetivos limitados; região projetada do braço majoritariamente dentro da imagem). Status: `ok` (≥ 0,6), `degradado` (0,3–0,6, overlay continua mas o jogo não pune) e `perdido` (< 0,3 por ≥ 5 quadros).

**Re-segmentação e correção de deriva:**
- Pedido ao SegWorker a cada `T_reseg = 2 s`, ou imediatamente quando o status cai para degradado. O prompt é uma caixa/pontos tirados da região projetada (SAM 2) ou texto (SAM 3).
- A resposta corresponde ao quadro do pedido *i*. Comparamos com a região de K projetada por H_i: **IoU < 0,7** ⇒ correção.
- Correção: a máscara determina bem a direção do eixo e a posição transversal (bordas), mas **não** o deslizamento ao longo do eixo. Por isso a correção ajusta rotação + translação transversal + escala transversal, e o deslizamento ao longo do eixo é corrigido só por casamento ORB/ECC contra K. Se nem assim houver concordância, o estado vai para PERDIDO.

**Relocalização (`relocalizer`)** — em PERDIDO, a cada 3 quadros: ORB (1000 pontos) no quadro (ou na ROI de uma detecção nova, se houver), casamento Hamming com teste de razão 0,75 contra os descritores de K, homografia RANSAC; aceita se inliers ≥ 30 e H plausível. Depois reinicializa o LK. Se não recuperar em `T_relock = 5 s`, a calibração é descartada (RF-38).

**Variante A2 — SAM 2 video predictor** (avaliação, Marco 5): propaga a máscara de forma assíncrona; a pose sai da geometria da máscara (como em B). Serve para comparar contra o LK.

### 7.2 Módulo B — YOLO open-vocabulary (opcional)

YOLOE-seg (texto "guitar") dá caixa **e máscara** a cada quadro (estimativa: 10–20 ms na 3060, a validar). A máscara passa pela mesma `mask_geometry`. Dois modos de pose:
- **B-quadro**: pose só pela geometria da máscara: eixo + bordas + duas âncoras ao longo do eixo (transição cabeça/braço e braço/corpo pelo perfil de largura) → 4 cantos → homografia K→atual. Sofre com o deslizamento ao longo do eixo (as âncoras são ruidosas), e é justamente isso que queremos medir.
- **B+track**: YOLOE só como localizador do quadro-chave/relocalização, com o mesmo tracker LK de A.

YOLO-World dá só caixa (sem orientação). Nesse caso a caixa vira prompt para SAM 2 ou ROI para o módulo C.

### 7.3 Módulo C — OpenCV clássico (opcional, *baseline*)

ROI = vizinhança da mão de notas (o braço está sob ela) ampliada ao longo da orientação dominante, ou o quadro todo. Etapas: CLAHE → Canny → `HoughLinesP`. As **cordas** são o maior agrupamento de segmentos longos quase paralelos (histograma de ângulos); a direção desse agrupamento é o eixo, e as cordas extremas dão as bordas. Os **trastes** são segmentos curtos aproximadamente perpendiculares ao eixo: projeção do gradiente ao longo de s → picos → posições de trastes, que podem ser ajustadas diretamente ao modelo 12-TET (dispensando a pestana). Esperado: frágil com fundo texturizado, *motion blur* e iluminação ruim. Também serve como **refino opcional** de A (encaixar os trastes previstos nos picos de gradiente).

### 7.4 Sem marcadores (ex-módulo D)

Por decisão do projeto, **nada é colado no violão**. Não há módulo de referência com marcadores; o *ground truth* da avaliação vem de anotação manual de quadros amostrados (evaluation §3). A numeração das seções foi mantida.

### 7.5 Posição recomendada e estimativa de inclinação (risco K1)

**Recomendação mostrada ao usuário (tela de ajuda + aviso automático):**
- Tampo do violão voltado para a câmera, paralelo ao plano da imagem (± 15°). **Não** inclinar o tampo para trás para "olhar o braço" (é o hábito natural, e comprime as cordas).
- Braço entre 0° e 35° acima da horizontal.
- Câmera de frente, na altura do braço, a **0,7–0,9 m**, de modo que o trecho nut → traste 12 ocupe a maior parte possível da largura do quadro (meta ≥ 50%) a 1280×720.
- Luz frontal difusa, sem contraluz (janela atrás do usuário). Fundo sem linhas fortes paralelas ao braço.

**Estimativa de inclinação** (`pose_quality.estimate_tilt`):
- Rotação no plano: ângulo do eixo na imagem atual (direto de H_t aplicada ao eixo de K).
- Fora do plano (depois da calibração): razão observada ρ = largura_px(g = 0,25) / L_px, comparada com a esperada ρ₀ = largura_física(g = 0,25)/L (aço: ≈ 47/648 ≈ 0,073; configurável).
  - ρ < ρ₀ indica **rolagem** do tampo para trás/frente (comprime a largura): ângulo ≈ arccos(ρ/ρ₀).
  - ρ > ρ₀ indica **guinada** (braço apontando para a câmera, encurta o comprimento): ângulo ≈ arccos(ρ₀/ρ).
  - Os dois efeitos se compensam parcialmente e não são separáveis com uma única razão; reportamos um **índice de compressão** |ln(ρ/ρ₀)| e avisamos quando ultrapassa ln(1/cos 40°) ≈ 0,27.
- Gradiente de perspectiva ao longo do braço: o parâmetro *c* da calibração projetiva (§8.2), que é zero quando o braço está paralelo ao plano da imagem.
- Antes da calibração: a mesma ideia com o comprimento da região estreita da máscara (≈ nut → junção com o corpo) em relação à largura.

### 7.6 Oclusão (risco K3)

- Máscara das mãos (casco convexo dos 21 pontos, dilatado em 15 px) excluída da detecção e do rastreamento de pontos.
- Pontos distribuídos pela cabeça, pelo espelho fora da mão e pela junção com o corpo. A homografia precisa de ≥ 4 pontos não colineares; exigimos inliers em pelo menos 2 das 3 regiões (cabeça, espelho, corpo) para status `ok`.
- O braço direito pode cobrir a junção com o corpo; a cabeça raramente fica oculta. Se a cabeça sair do quadro, a qualidade cai (aviso de enquadramento).

### 7.7 Webcam: exposição e FPS (risco K6)

- Ao abrir a câmera: MJPG, 1280×720, 30 FPS. Tentativa de `CAP_PROP_AUTO_EXPOSURE` manual e `CAP_PROP_EXPOSURE` fixo (no DSHOW os valores são log₂ em segundos, por exemplo −6 ≈ 1/64 s; −7 ≈ 1/128 s, que reduz o *blur*), ganho fixo e WB fixo. Ler de volta e registrar.
- O FPS efetivo é medido (média móvel); se < 20, aviso na tela ("pouca luz / auto exposição").
- Se o FPS medido for baixo, o juiz amplia automaticamente o *debounce* e as janelas proporcionalmente ao período do quadro (§9.4).

## 8. Calibração

### 8.1 Modelo dos trastes

Com a escala L e o nut em u = 0, o traste n fica em

  u(n) = L · f(n),  f(n) = 1 − 2^(−n/12),  f(12) = 0,5.

Trabalhamos com g = u/L (fração da escala). O centro da casa n (entre os trastes n−1 e n) e a posição típica do dedo ficam em

  g_dedo(n) = f(n−1) + β · (f(n) − f(n−1)),  β ∈ [0, 1].

β modela o **deslocamento do dedo em relação ao traste**: o dedo fica atrás do traste, mais perto dele que do traste anterior. Valor padrão β = 0,75 (a estimar em evaluation §5.3). Sensibilidade: um erro de 0,1 em β desloca g_dedo(1) em 0,0056 e g_dedo(12) em 0,003, ou seja, cerca de 4 mm e 2 mm com escala de 648 mm. O efeito é pequeno.

Valores (β = 0,75): g_dedo(1) = 0,04210 · g_dedo(5) = 0,23970 · g_dedo(12) = 0,49257.

### 8.2 Do eixo da imagem para g: modelos 2 pontos e 3 pontos

A medida de cada pestana é a coordenada s_i (em K) do ponto em que o indicador cruza a linha central do braço (§8.4).

**Modelo afim (2 pontos: casas 1 e 12)** — a ideia original:

  s = s₀ + L_px · g
  L_px = (s₁₂ − s₁) / (g_dedo(12) − g_dedo(1)),  s₀ = s₁ − L_px · g_dedo(1).

O sinal de L_px define o sentido do braço (e é verificado contra `nut_side_hint`).

**Problema de perspectiva.** A projeção de um segmento num plano inclinado não é afim. Com o trecho nut → 12 de comprimento ℓ ≈ 325 mm a uma distância Z da câmera e guinada θ, o erro máximo do modelo afim (no meio do trecho, em unidades físicas) é aproximadamente

  e_max ≈ (ℓ²/4) · sin θ / Z.

Para Z = 0,8 m: θ = 10° → ~6 mm; θ = 15° → ~9 mm; θ = 30° → ~17 mm, enquanto a casa 6 mede ~27 mm. **Uma guinada de 15–30° já tira o meio do braço de 1/3 a 2/3 de casa do lugar.** Por isso o design propõe:

**Modelo projetivo 1D (3 pontos: casas 1, 5 e 12)** — homografia da reta:

  s = (a·g + b) / (c·g + 1)

Cada medida (g_i, s_i) dá uma equação **linear** nas incógnitas: a·g_i + b − c·g_i·s_i = s_i. Com 3 medidas, o sistema 3×3 tem solução direta (com mais medidas, mínimos quadrados). Nut: s₀ = b. Restrição: c·g + 1 > 0 em g ∈ [0; 0,55]. A casa 5 foi escolhida porque o erro do modelo afim é máximo perto de g ≈ 0,25 (casa ~5), e a pestana na casa 5 é confortável.

**Modelo 2 pontos + afunilamento** (alternativa sem 3ª pestana, a avaliar): sob a homografia, a ampliação local transversal é proporcional a 1/(c·g + 1). Com as larguras medidas nas bordas de K em g_dedo(1) e g_dedo(12) e a razão física de largura conhecida (aço: ≈ 43 → 54 mm; configurável), estimamos c. O resultado depende da qualidade das bordas e da precisão do afunilamento informado.

**Decisão:** padrão **2 pontos (casas 1 → 12)**, como na ideia original. Com o violão de frente para a câmera (posição recomendada, §7.5) o modelo afim basta: guinada ≤ ~10° dá erro ≤ ~6 mm, cerca de 1/4 de casa no meio do braço, dentro da meta RNF-04. O modo 3 pontos (pestana extra na casa 5) e o 2p+afunilamento ficam como opções de configuração (`calib.mode`), e evaluation §5.3 mede quanto ganham quando o violão gira.

Inverso (imagem → g): afim g = (s − s₀)/L_px; projetivo g = (s − b)/(a − c·s).

### 8.3 Modelo das cordas e ponto de contato

- Em K, as bordas do braço são retas t_bord(s) (bordões) e t_prim(s) (primas). Normalizamos w = (t − t_bord(s)) / (t_prim(s) − t_bord(s)).
- Cordas com margem m (fração da largura entre a borda e a corda extrema; padrão m = 0,09; num nut de 43 mm, isso dá ~35 mm entre as cordas extremas): w_k = m + (6 − k)/5 · (1 − 2m). Corda contínua: k̂ = 6 − 5·(w − m)/(1 − 2m).
- Refino opcional: em K, o perfil de intensidade transversal pode localizar as cordas reais (picos claros/escuros) e ajustar m e o deslocamento. Feito uma vez, no quadro-chave.
- **Ponto de contato do dedo**: a ponta do MediaPipe (pontos 8/12/16/20) marca o extremo do dedo; o contato com a corda fica um pouco atrás. p_c = p_ponta + γ·(p_DIP − p_ponta), com γ padrão 0,2, ajustado na T4.5.

### 8.4 Detecção de pestana

Por quadro, para a mão de **notas** (pontos do MediaPipe em coordenadas do braço via H_t⁻¹):
1. **Indicador estendido**: ângulos em PIP (6) e DIP (7) ≥ 150°. Na pestana o indicador fica reto, ao contrário dos outros dedos, que ficam dobrados. Os outros dedos não são verificados (podem estar dobrados ou formando um acorde).
2. **Perpendicular ao eixo**: reta ajustada a 6–7–8 com ângulo em relação à direção transversal do braço ≤ 25°.
3. **Cobertura**: o segmento 6→8 projetado em w cobre ≥ 0,5 da largura do braço (o indicador atravessa as cordas).
4. **Sobre o braço**: o cruzamento com a linha central (w = 0,5) cai dentro da extensão do braço.
5. **Medida**: s_i = coordenada s do cruzamento da reta 6–7–8 com a linha central. Usar a reta do dedo, e não só a ponta, dá robustez: o eixo do indicador na pestana é praticamente paralelo ao traste.
6. **Estabilidade**: janela deslizante de 1,0 s (configurável) com ≥ 80% dos quadros válidos e desvio-padrão de s_i ≤ 15% do espaçamento estimado da casa (antes de calibrar, usamos 3% da extensão do braço). A medida final é a **mediana**.

**Risco K4** (MediaPipe falha com os dedos dobrados sobre o braço): a confiança do MediaPipe e a taxa de quadros válidos são registradas. Se a captura não acontecer em 10 s, a tela oferece a **calibração manual** (cliques no nut e no traste 12, e opcionalmente no 5). A taxa de sucesso é medida em evaluation §5.4.

### 8.5 Validação da calibração

Rejeitar e pedir para refazer quando:
- s₁₂ − s₁ tiver sinal oposto ao de `nut_side_hint` (quando disponível), ou |s₁₂ − s₁| < 3·largura_braço_px. Esperado (aço): (0,4926 − 0,0421)·648 ≈ 292 mm contra ~48 mm de largura média, razão ≈ 6,1.
- A razão (|s₁₂ − s₁| / largura) estiver fora de [3; 10].
- No modelo 3p: c·g + 1 ≤ 0 em [0; 0,55], ou |c| > c_max (perspectiva implausível).
- O nut previsto cair fora da extensão do braço (com margem).

Depois de validada, a calibração mostra os trastes sobre o vídeo e pede ENTER (aceitar) ou R (refazer).

### 8.6 Re-calibração

- Tecla **C** a qualquer momento.
- Automática quando PERDIDO dura mais que `T_relock`, porque a relocalização falhou e a homografia para K deixou de ser confiável.
- O que sobrevive à re-calibração: β, m, γ e as preferências do usuário. O L_px e o nut dependem de K e são refeitos.

## 9. Lógica do jogo

### 9.1 Formato do chart

```json
{
  "versao": 1,
  "titulo": "Pentatônica menor de Lá — casa 5 (sobe e desce)",
  "artista": "exercício",
  "bpm": 60,
  "offset_ms": 0,
  "afinacao": ["E2", "A2", "D3", "G3", "B3", "E4"],
  "notas": [
    { "t": 4.0, "corda": 6, "casa": 5 },
    { "t": 5.0, "corda": 6, "casa": 8 },
    { "t": 6.0, "corda": 5, "casa": 5 }
  ]
}
```
(trecho; a sequência completa está em §9.1.1)

Regras de validação: `versao` = 1; `t` ≥ 0 e não decrescente; `corda` ∈ 1..6; `casa` ∈ 1..12; `dur` ≥ 0 opcional (padrão 0, apenas visual). `bpm` define o metrônomo visual e a contagem de 4 tempos. `offset_ms` desloca todas as notas. Erros indicam o índice da nota.

#### 9.1.1 Chart de exemplo: pentatônica menor de Lá na casa 5

O desenho enviado é a **pentatônica menor de Mi na posição aberta** (cordas soltas + casas 2/3). Transposto para começar na casa 5 (as cordas soltas viram casa 5), vira a **pentatônica menor de Lá, "box 1"** (notas Lá, Dó, Ré, Mi, Sol):

| Corda | 6 (Mi) | 5 (Lá) | 4 (Ré) | 3 (Sol) | 2 (Si) | 1 (Mi) |
|---|---|---|---|---|---|---|
| Casas | 5, 8 | 5, 7 | 5, 7 | 5, 7 | 5, 8 | 5, 8 |
| Notas | Lá, Dó | Ré, Mi | Sol, Lá | Dó, Ré | Mi, Sol | Lá, Dó |

Dedilhado sugerido: indicador na casa 5, anelar na 7, mínimo na 8.

Sequência (uma nota por tempo, 23 notas; sobe e desce sem repetir a nota do topo), no formato (corda, casa):

- **Sobe**: (6,5) (6,8) (5,5) (5,7) (4,5) (4,7) (3,5) (3,7) (2,5) (2,8) (1,5) (1,8)
- **Desce**: (1,5) (2,8) (2,5) (3,7) (3,5) (4,7) (4,5) (5,7) (5,5) (6,8) (6,5)

Tempos: nota *i* (i = 0..22) em t_i = 4·T + i·T, com T = 60/bpm (4 tempos de contagem antes). Dois arquivos:
- `charts/pentatonica_la_casa5_60bpm.json`: T = 1,0 s, notas de 4,0 s a 26,0 s.
- `charts/pentatonica_la_casa5_90bpm.json`: T ≈ 0,667 s, notas de 2,667 s a 17,333 s.

Observação: no modo "zonas", (6,5) e (5,5) caem na mesma zona (graves), então trocar uma pela outra conta como acerto. O modo "exata" diferencia.

### 9.2 Relógio

- Relógio mestre: `perf_counter` desde o início da partida. Metrônomo **visual**: pulso no HUD a cada tempo do `bpm` e contagem de 4 tempos antes da primeira nota.
- Tempo de jogo de um quadro de vídeo: `t_jogo(frame) = (frame.ts − latencia_camera) − t_inicio + offset_usuario`.
- Pausa (RF-56): congela o relógio; ao retomar, contagem de 3 s.

### 9.3 Visualização das notas

- **Anel de aproximação** (padrão): na célula (k, n), a bolinha aparece em t_nota − T_lead (T_lead = 1,5 s) com um anel que se fecha linearmente até o raio da bolinha em t_nota. Cor por corda. Raio da bolinha = 0,4·min(espaçamento entre cordas, largura da casa) **no ponto projetado**, o que acompanha a perspectiva.
- **Deslizante** (alternativo): a bolinha percorre a corda k a partir do lado do corpo (g = 0,55) até a casa n, chegando em t_nota.
- Feedback: anel verde (perfeito) ou amarelo (bom), "X" vermelho (erro), texto flutuante com pontos e combo.
- Faixa de próximas notas (opcional) no HUD.

### 9.4 Detecção de acerto (somente visão)

**Contatos por quadro**: para cada dedo 2–5 da mão de notas, ponto de contato → (n̂, k̂). O dedo está **na célula (k, n)** se:
- casa: n − 1 − ε_n < n̂ ≤ n + ε_n, com ε_n = 0,15 casa (padrão);
- corda, modo **zonas** (padrão): k̂ pertence à mesma zona de k (graves {6,5}, médias {4,3}, agudas {2,1}), com tolerância de ±0,5 corda nas fronteiras externas;
- corda, modo **exata**: |k̂ − k| ≤ ε_k, com ε_k = 0,5 (ajustável).

**Presença com debounce**: presença(t) é verdadeira quando o dedo está na célula em ≥ `k_deb` quadros consecutivos (padrão 2, ≈ 67 ms a 30 FPS; se o FPS for menor, `k_deb` cai para manter ≥ ~60 ms com mínimo de 1).

**Julgamento** (a visão vê *onde* o dedo está, não *quando* a corda é tocada, por isso o critério é "dedo no lugar certo no momento certo"):
- Janela: [t_nota − W_antes, t_nota + W_depois], padrão W_antes = 600 ms, W_depois = 200 ms.
- **Perfeito**: presença verdadeira no quadro mais próximo de t_nota (± 1 quadro).
- **Bom**: presença em algum quadro da janela, mas não em t_nota (chegou atrasado até +200 ms, ou chegou cedo e saiu).
- **Erro**: nenhuma presença na janela.
- Notas repetidas na mesma célula: o dedo parado conta para cada uma (consequência natural da regra, o que é coerente com o violão real).
- Cada nota é julgada uma vez, no fechamento da janela (ou antes, quando já for perfeito).

### 9.5 Pontuação

| Julgamento | Pontos base | Combo |
|---|---|---|
| Perfeito | 100 | +1 |
| Bom | 50 | +1 |
| Erro | 0 | zera |

Multiplicador = min(1 + ⌊combo/10⌋, 4); pontos = base × multiplicador. Precisão = (perfeitos + 0,5·bons)/total. Conceito: S ≥ 95%, A ≥ 85%, B ≥ 70%, C ≥ 50%, D abaixo disso. A tela final mostra tudo isso e também o combo máximo; os resultados vão para `results/<data>_<chart>.json`, com o log de cada julgamento (t_nota, Δ, célula alvo, célula observada, modo).

### 9.6 Viabilidade de distinguir cordas (risco K2)

Webcam com campo horizontal de ~70°, 1280 px de largura:

| Distância | px/mm | Corda–corda (aço, nut → casa 12) | Casa 1 / casa 6 / casa 12 |
|---|---|---|---|
| 0,7 m | 1,31 | 9 → 12 px | 48 / 36 / 25 px |
| 1,0 m | 0,91 | 6,4 → 8,4 px | 33 / 25 / 18 px |
| 1,2 m | 0,76 | 5,3 → 7 px | 28 / 21 / 15 px |

A 640×480 os valores caem pela metade. O erro esperado da ponta do dedo no MediaPipe é de alguns pixels, mais um viés de contato (a medir na T4.5). **Conclusão:** casas são distinguíveis com folga a ≤ 1 m; cordas individuais ficam no limite, mesmo a 0,7 m. Daí as decisões: modo "zonas" como padrão, câmera a 0,7–0,9 m e 720p. Sem áudio, a corda depende só da visão; o modo "exata" fica experimental até o resultado da T4.5. As tolerâncias finais (ε_n, ε_k, γ) saem do experimento da T4.5 (evaluation §5.5).

## 10. Estados

### 10.1 Estado do rastreamento/calibração (app)

```
                     ┌───────────────────────────────────────────────┐
                     ▼                                               │ T_relock expirou
┌──────────────────────────────────┐  calibração ok  ┌────────────┐  │ (descarta calib.)
│ SEM_CALIBRACAO                   │ ──────────────► │ RASTREANDO │ ─┼──┐
│  · procurando braço              │ ◄── tecla C ─── │            │  │  │ qualidade < limiar
│  · braço ok (overlay de eixo)    │                 └────────────┘  │  │ por N quadros
└──────────────────────────────────┘                     ▲           │  ▼
     │ braço ok + tecla/automático                       │ reloc. ok ┌─────────┐
     ▼                                                   └────────── │ PERDIDO │
┌──────────────────────────────────┐                                 └─────────┘
│ CALIBRANDO                       │  perda do braço → SEM_CALIBRACAO.procurando
│  casa1 → casa12 [→ casa5 se 3p] →│  timeout 10 s → oferece calibração manual
│  validar → confirmar (ENTER/R)   │  ESC → SEM_CALIBRACAO
└──────────────────────────────────┘
```

| De | Evento | Para | Ação |
|---|---|---|---|
| SEM_CALIBRACAO.procurando | localizador encontra o braço (conf ≥ 0,5) | SEM_CALIBRACAO.braço_ok | cria K, inicia tracker |
| SEM_CALIBRACAO.braço_ok | pose perdida | SEM_CALIBRACAO.procurando | pedidos ao SegWorker a cada 0,5 s |
| SEM_CALIBRACAO.braço_ok | tecla ESPAÇO (ou calibração carregada) | CALIBRANDO / RASTREANDO | |
| CALIBRANDO | captura i válida | próxima etapa | sinal visual (flash + check) |
| CALIBRANDO | validação ok + ENTER | RASTREANDO | grava a calibração em K |
| CALIBRANDO | validação falha / R | CALIBRANDO (início) | mensagem |
| RASTREANDO | status `perdido` | PERDIDO | esconde overlay do braço, pausa o jogo |
| PERDIDO | relocalização ok | RASTREANDO | reinicia LK, retoma com contagem |
| PERDIDO | T_relock expirou | SEM_CALIBRACAO.procurando | descarta calibração |
| qualquer | tecla C | CALIBRANDO | |

### 10.2 Estado do jogo

`MENU → (seleciona chart; exige RASTREANDO) → CONTAGEM (3 s) → JOGANDO ⇄ PAUSADO (por PERDIDO ou tecla P) → FIM (tela final) → MENU`.

## 11. Decisões e alternativas descartadas

| Decisão | Alternativas descartadas | Motivo |
|---|---|---|
| Homografia K → quadro atual com LK + RANSAC | Re-detectar o braço em todo quadro com SAM | SAM 3 é lento demais na 3060 para ≥ 20 FPS junto com o resto; detecção por quadro tem *jitter* e não fixa o deslizamento ao longo do eixo. |
| | Composição de transformações quadro a quadro | Acumula deriva; estimar sempre K → atual limita isso. |
| | `cv2.TrackerCSRT`/KCF (caixa) | Dá só caixa alinhada aos eixos, sem perspectiva nem orientação; serve no máximo para ROI. |
| Geometria guardada em K (referencial do braço) | Grade em coordenadas da imagem | Violaria R2; quebra com o movimento. |
| Papel das mãos por geometria | Só o rótulo de lateralidade do MediaPipe | O rótulo assume imagem espelhada (selfie) e erra com as mãos cruzadas ou parciais. |
| Processar a imagem **não espelhada**; espelhar só a exibição | Espelhar a entrada | Mantém a geometria coerente com o mundo e com as anotações; o espelhamento vira só exibição. |
| SegWorker assíncrono (thread) | Processo separado | Mais simples; PyTorch e OpenCV liberam o GIL nas operações pesadas. Se o profiler mostrar disputa, migramos para `multiprocessing` (fica anotado). |
| Pestana medida pela **reta do indicador** | Ponta do indicador | Na pestana a ponta fica fora do braço (além da corda 6 ou 1); a reta é paralela ao traste. |
| Modelo afim de 2 pontos como padrão; projetivo 1D (3 pontos) como opção | Projetivo como padrão | Com o violão de frente, o erro do afim fica em ~1/4 de casa no pior ponto; a 3ª pestana só compensa com 15–30° de guinada (§8.2), e a avaliação vai medir isso. |
| | Pose 3D completa (PnP) do violão | Exige pontos 3D conhecidos no violão (marcadores), o que foi descartado. |
| Julgamento por presença com janelas assimétricas | Janela simétrica em torno da "chegada" do dedo | Na prática o dedo chega antes da palhetada; a visão não vê a palhetada. |
| Modo "zonas de cordas" como padrão | Corda exata como padrão | Resolução insuficiente (§9.7). |
| Só imagens nesta versão | Pitch/onset do microfone como reforço | Decisão do usuário. A arquitetura não impede adicionar áudio depois (o juiz é uma interface). |
| *Ground truth* por anotação manual | Marcadores ArUco no violão | Decisão do usuário: nada colado no violão. |
| Sem treino de modelos | Detector de trastes treinado, *fine-tuning* de YOLO | Proibido pelo enunciado (R1). |
| — | MediaPipe Holistic/Pose | Mais pesado e desnecessário para o objetivo. |
| — | Câmera de profundidade, estéreo | Fora do hardware disponível. |

## 12. Configuração (chaves principais de `configs/default.yaml`)

```yaml
camera:   { index: 0, backend: dshow, width: 1280, height: 720, fps: 30, fourcc: MJPG,
            auto_exposure: false, exposure: -6, gain: null, latency_ms: 70 }
display:  { mirror: true, debug: false }
hands:    { max_hands: 2, min_conf: 0.5, handedness_mirrored_input: false, fretting: left }
neck:     { locator: sam, tracker: lk, reseg_period_s: 2.0, iou_reset: 0.7,
            min_inliers: 25, lost_frames: 5, relock_timeout_s: 5.0,
            sam: { version: 3, prompts: ["guitar neck", "fretboard", "guitar"] } }
calib:    { mode: 2p, frets: [1, 12], beta: 0.75, hold_s: 1.0, scale_mm: 648,   # mode: 2p | 2p_taper | 3p
            width_nut_mm: 43, width_12_mm: 54, string_margin: 0.09, bass_edge: top }
game:     { lead_s: 1.5, style: ring, w_before_ms: 600, w_after_ms: 200,
            eps_fret: 0.15, string_mode: zones, eps_string: 0.5, debounce_frames: 2,
            contact_gamma: 0.2, user_offset_ms: 0, count_in_beats: 4 }
log:      { jsonl: true, dir: logs/ }
```

## 13. Plano de testes

| Nível | O que | Como | Requisitos |
|---|---|---|---|
| **Unitário** | `fret_model` (f(n), inversos), `solver` 2p/3p, `string_model`, conversões de coordenadas, `chart` (validação), `judge` (linhas do tempo sintéticas de presença), `score`, `filters` | `pytest`, dados sintéticos; tolerâncias numéricas explícitas | RNF-11 |
| **Sintético de geometria** | Solver sob perspectiva: gerar s_i a partir de homografias 1D conhecidas (θ de 0 a 40°) + ruído σ de 0 a 6 px → erro dos trastes 1–12 para 2p, 2p+afunilamento e 3p | Script + teste com limiar | RF-32–34 |
| **Sintético de tracking** | Deformar um quadro-chave real por sequências conhecidas de homografias (+ ruído, *blur*, oclusão retangular simulando a mão) → erro de H recuperada (erro de canto, em px) | Teste com limiar e gráfico | RF-23–27 |
| **Integração (replay)** | Gravações curtas de referência (parado, movimento lento, oclusão, perda e retorno, sessão de calibração, partida curta) → executar o pipeline, comparar com saídas de referência (estado por quadro, número de julgamentos) e verificar determinismo | `pytest -m replay` (fora da CI padrão por causa dos arquivos grandes) | RF-03, RNF-08, CA-03 |
| **Desempenho** | Benchmark por estágio em 60 s de replay e ao vivo; relatório P50/P95 | `python -m ghirl.eval.bench` | RNF-01, RNF-02 |
| **Demo manual por marco** | Checklist de demonstração no fim de cada marco ([tasks.md](tasks.md)) | Vídeo curto gravado como evidência | todos os CA |
| **Avaliação experimental** | [evaluation.md](evaluation.md) | Scripts `ghirl.eval.*` | RF-70–73 |
