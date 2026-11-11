# Guitar Hero IRL — Tarefas

> Versão 0.2 (revisada: só imagens, sem ArUco, violão de aço, calibração de 2 pontos) · 2026-11-11
> Requisitos: [requirements.md](requirements.md) · Design: [design.md](design.md) · Avaliação: [evaluation.md](evaluation.md)

**Como usar este arquivo**
- Implementação só depois da aprovação da spec, **uma task por vez**, na ordem (respeitando as dependências).
- Ao concluir, marcar `[x]` e anotar a data e uma linha de evidência (teste, vídeo, número medido).
- **Pronto** = critério da task atendido + testes da task passando + `ruff` limpo.
- Tasks marcadas **(opcional)** podem ser puladas se faltar tempo, sem quebrar os marcos seguintes.
- Cada marco termina numa task **Demo**, com checklist e um vídeo curto como evidência.

---

## Marco 1 — Webcam + overlay fixo, ponta a ponta

Demonstração: o app abre a webcam, desenha uma grade fixa de bolinhas e os pontos das mãos, e mostra FPS e tempos por estágio; uma sessão pode ser gravada e reproduzida.

- [x] **T1.1 — Ambiente e esqueleto do projeto** · concluída em 2026-11-11
  - Evidência: `python -m ghirl.utils.check_env` → Python 3.12.14, torch 2.11.0+cu128, CUDA disponível, RTX 3060 12 GiB, `OK`; `pytest` passa; `ruff` limpo. Nota: usamos `opencv-contrib-python` (vem com o mediapipe) e **não** `opencv-python` (conflito no `cv2`; há teste para isso).
  - Depende de: —
  - Fazer: venv Python 3.12; `requirements.txt` com versões fixadas (opencv-contrib-python, mediapipe, numpy, pyyaml, pytest, ruff; torch+CUDA instalado à parte); estrutura de pacotes `ghirl/` (design §3); README de instalação; `pytest` com um teste trivial.
  - Pronto quando: instalação limpa segue o README; `pytest` passa; um script imprime as versões e `torch.cuda.is_available() == True` com o nome da GPU.
  - Ref.: RNF-09, RNF-11, K14

- [x] **T1.2 — Configuração** · concluída em 2026-11-11
  - Evidência: `ghirl/app/config.py` + `configs/default.yaml`; `tests/test_config.py` (21 casos: padrões = YAML, arquivo, `--set`, chaves desconhecidas, tipos e valores inválidos).
  - Depende de: T1.1
  - Fazer: `configs/default.yaml` (design §12), carregamento em dataclasses, sobrescrita por CLI (`--set chave=valor`).
  - Pronto quando: teste unitário cobre o carregamento, a sobrescrita e o erro para chave inexistente.
  - Ref.: RF-80

- [x] **T1.3 — Captura da webcam em thread** · concluída em 2026-11-11
  - Depende de: T1.2
  - Fazer: `CameraSource` (thread, último quadro, `ts` com `perf_counter`, índice); aplicar resolução/FPS/MJPG/exposição manual com leitura de volta; medir o FPS real; comparar DSHOW × MSMF.
  - Pronto quando: script de diagnóstico imprime as propriedades aplicadas e o FPS real em 10 s, com auto-exposição ligada e desligada; resultado anotado aqui.
  - Ref.: RF-01, RF-02, CA-01, CA-02, K6
  - Evidência — código e testes: `ghirl/capture/camera.py`, `diagnose.py`, `tests/test_camera.py`. Resultados (`python -m ghirl.capture.diagnose`):

    | Backend | Auto-exposição | Resolução | FOURCC | FPS real | Brilho médio |
    |---|---|---|---|---|---|
    | DSHOW | sim | 1280×720 | MJPG | 29,9 | 160 |
    | DSHOW | não (−6) | 1280×720 | MJPG | 29,8 | 170 |
    | MSMF | sim | 1280×720 | (não informa) | 30,4 | 159 |
    | MSMF | não | 1280×720 | (não informa) | 30,6 | 175 (exposição ignorada) |

    - No DSHOW a ordem importa: largura/altura/FPS **antes** do FOURCC; ao contrário, o formato volta para YUY2 e cai para ~7 FPS.
    - O MSMF não aplica exposição manual; o DSHOW aplica. Decisão provisória: **DSHOW**.
    - Varredura de exposição no DSHOW (uma abertura de câmera por valor, luz ambiente do quarto):

      | Exposição | ≈ tempo | FPS real | Brilho médio |
      |---|---|---|---|
      | −6 (padrão) | 1/64 s | 29,9 | 152 |
      | −7 | 1/128 s | 29,9 | 108 |
      | −8 | 1/256 s | 29,7 | 69 |
      | −4 | 1/16 s | ~16 | saturado |

    - Decisão: padrão **−6**; usar **−7** com boa luz frontal (menos *blur* em movimento). −4/−5 limitam o FPS.
    - Cuidado: trocar a exposição várias vezes com a câmera aberta travou a webcam (quadros pretos a 1 FPS, MSMF sem quadros) até ela ser reconectada. O código aplica as propriedades uma vez só, na abertura.

- [x] **T1.4 — Gravação e replay** · concluída em 2026-11-11
  - Evidência: `python -m ghirl.capture.recorder --out data/rec/t14_teste --seconds 20` → 596 quadros, 20,2 s, 29,5 FPS, 12 MB (MJPG q95); replay sequencial devolve os mesmos 596 quadros com `ts` idênticos. `tests/test_record_replay.py`: contagem/ts, determinismo, FFV1 sem perdas, modo tempo real com pulo de quadros, loop, vídeo comum sem timestamps.
  - Depende de: T1.3
  - Fazer: `Recorder` (vídeo sem perdas ou MJPG de alta qualidade + `timestamps.json` + metadados da câmera); `ReplaySource` com a mesma interface `FrameSource` (modo tempo real e modo "o mais rápido possível").
  - Pronto quando: gravar 20 s e reproduzir; teste confirma o mesmo número de quadros e os mesmos `ts`.
  - Ref.: RF-03, RF-71

- [x] **T1.5 — Laço principal + render com overlay fixo + HUD** · concluída em 2026-11-11
  - Evidência: `python -m ghirl.app.main` (webcam), `--replay <pasta>` e `--record <pasta>`; teclas ESC/Q, D, M, H. Testes: a imagem processada não é alterada; exibição espelhada = `flip` da não espelhada (overlay desenhado no mundo, HUD depois do espelho). Conferência visual da janela fica para a demo T1.8.
  - Depende de: T1.3
  - Fazer: `app/main.py` com o laço (design §6, só as etapas 0, 8 e 9 por enquanto); grade fixa de bolinhas 6×12; HUD (FPS de captura e de processamento); espelhamento só na exibição; teclas (ESC, D = depuração).
  - Pronto quando: a grade aparece sobre o vídeo ao vivo e em replay; com o espelho ligado, a imagem exibida espelha e a processada não.
  - Ref.: RF-04, RF-40, RF-43, CA-04 (parte)

- [x] **T1.6 — Profiler e log JSONL** · concluída em 2026-11-11
  - Evidência: tabela ao sair (média/P50/P95/máx por estágio). Webcam, 300 quadros: total 3,2 ms/quadro (overlay 2,1 + render 1,1), 1 quadro descartado. Replay, 596 quadros: 3,5 ms. JSONL em `logs/` lido por `tests/test_app_loop.py`.
  - Depende de: T1.5
  - Fazer: `utils/profiler.py` (cronômetro por estágio, P50/P95), log JSONL por quadro (índice, ts, estado, tempos); relatório ao sair.
  - Pronto quando: ao fechar o app, aparece a tabela de tempos por estágio; o JSONL é válido e lido por um teste.
  - Ref.: RF-70, RNF-01

- [ ] **T1.7 — Mãos com MediaPipe + verificação de lateralidade**
  - Depende de: T1.5
  - Fazer: `HandLandmarker` em modo VIDEO; desenhar os 21 pontos; máscara das mãos (casco convexo dilatado); **experimento**: com imagem espelhada e não espelhada, levantar só a mão esquerda e registrar o rótulo retornado; definir `handedness_mirrored_input`; papel provisório pelo rótulo corrigido.
  - Pronto quando: tabela "entrada espelhada? × mão real → rótulo" preenchida aqui; com a configuração correta, a mão esquerda real é rotulada "notas" em ≥ 95% de 300 quadros.
  - Ref.: RF-10, RF-11 (provisório), RF-13, K5

- [ ] **T1.8 — Demo do Marco 1**
  - Depende de: T1.4, T1.6, T1.7
  - Checklist: vídeo ao vivo + grade fixa + mãos + HUD; FPS médio ≥ 20 com MediaPipe ligado (anotar o número); gravação e replay da mesma sessão funcionando.
  - Ref.: CA-01, CA-03 (parte), RNF-01

---

## Marco 2 — Localização do braço (módulo A) + eixo

Demonstração: o app encontra o braço sozinho, desenha o eixo e as bordas e segue o violão em movimento; quando o violão sai de cena, o estado vai para PERDIDO e volta quando ele reaparece.

- [ ] **T2.1 — Tipos, interfaces e utilitários de geometria**
  - Depende de: T1.2
  - Fazer: dataclasses e Protocols de design §4 (`Frame`, `Hand`, `NeckDetection`, `NeckPose`, `Keyframe`, `NeckLocator`, `NeckTracker`); `utils/geometry.py` (aplicar homografia, quadro (s, t) a partir de eixo, interseção de retas, projeção ponto→reta).
  - Pronto quando: testes unitários de ida e volta (imagem ↔ K ↔ (s, t)) com erro < 1e-6.
  - Ref.: RNF-07, RNF-11

- [ ] **T2.2 — Instalação e benchmark de SAM 3 / SAM 2.1 na RTX 3060**
  - Depende de: T1.1
  - Fazer: instalar SAM 3 (repositório oficial ou Ultralytics) e SAM 2.1 (tiny/small); rodar em 5 quadros gravados com o violão; testar os prompts "guitar neck", "fretboard" e "guitar"; medir latência (aquecido) e VRAM.
  - Pronto quando: tabela "modelo × prompt → latência, VRAM, máscara correta? (s/n)" anotada aqui; **decisão** registrada (SAM 3 texto ou SAM 2 + clique/caixa).
  - Ref.: RF-20, K7, K8

- [ ] **T2.3 — `mask_geometry`: isolar o braço e estimar eixo e bordas**
  - Depende de: T2.1, T2.2
  - Fazer: PCA, perfil de largura W(s), isolamento do trecho estreito, `nut_side_hint`, ajuste robusto das duas bordas, eixo bissetor, largura ao longo de s (design §7.1).
  - Pronto quando: em 10 quadros anotados à mão (eixo), erro de ângulo ≤ 5° e erro de borda ≤ 5 px em ≥ 9 deles; teste com máscaras sintéticas (retângulo + elipse) passa.
  - Ref.: RF-21, RF-22, CA-20

- [ ] **T2.4 — SegWorker assíncrono + `SamLocator`**
  - Depende de: T2.3
  - Fazer: thread com fila de tamanho 1 (pedido mais recente vence), resposta com `frame_index`; `SamLocator` implementa `NeckLocator`; inicialização do quadro-chave K.
  - Pronto quando: com o SAM rodando, o laço principal mantém o FPS (nenhum quadro do laço > 100 ms por causa do SAM, verificado no profiler); eixo e bordas aparecem em ≤ 5 s após a abertura.
  - Ref.: RF-20, RNF-06, CA-20

- [ ] **T2.5 — Rastreador LK + homografia + qualidade**
  - Depende de: T2.4, T1.7
  - Fazer: `feature_tracker` (Shi-Tomasi na região rastreável menos a máscara das mãos, LK com verificação ida-e-volta, H de K → atual via RANSAC, reposição de pontos); `pose_quality` (design §7.1, §7.6).
  - Pronto quando: teste sintético (quadro-chave deformado por homografias conhecidas + oclusão retangular) com erro médio de canto ≤ 2 px; ao vivo, o overlay do eixo acompanha um movimento lento; nenhum ponto rastreado sobre a mão.
  - Ref.: RF-23, RF-24, RF-13, CA-11, CA-21

- [ ] **T2.6 — `NeckEstimator`: re-segmentação, deriva e perda**
  - Depende de: T2.5
  - Fazer: política de re-segmentação (período e queda de qualidade), comparação por IoU no quadro do pedido, correção transversal, status `ok`/`degradado`/`perdido`.
  - Pronto quando: em replay de 2 min com movimento, a deriva visual fica controlada (IoU médio ≥ 0,8 contra o SAM); tirar o violão de cena leva a `perdido` em ≤ 5 quadros.
  - Ref.: RF-25, RF-26

- [ ] **T2.7 — Relocalização por ORB**
  - Depende de: T2.6
  - Fazer: descritores ORB em K; tentativa a cada 3 quadros em PERDIDO; aceitar por inliers e plausibilidade; reinicializar o LK.
  - Pronto quando: em replay "sai de cena 2 s e volta", o estado volta a `ok` em ≤ 3 s em ≥ 4 de 5 casos.
  - Ref.: RF-27, CA-22 (parte sem calibração)

- [ ] **T2.8 — Papéis das mãos por geometria**
  - Depende de: T2.5
  - Fazer: `hands/roles.py`: mão de notas = a que está mais próxima/sobreposta ao braço e do lado do nut; rótulo do MediaPipe como desempate; histerese para não trocar papéis por um quadro.
  - Pronto quando: ≥ 95% de acerto em 300 quadros anotados, com e sem espelhamento.
  - Ref.: RF-11, CA-10

- [ ] **T2.9 — Overlay de depuração do braço + estimativa de inclinação**
  - Depende de: T2.6
  - Fazer: desenhar eixo, bordas, pontos (inliers/outliers), máscara, quadro-chave em miniatura, status; índice de compressão (design §7.5, versão pré-calibração); tela de ajuda com a posição recomendada.
  - Pronto quando: o aviso de posição ruim aparece ao girar o violão para o perfil e some na posição recomendada.
  - Ref.: RF-28, RF-41, CA-23

- [ ] **T2.10 — Demo do Marco 2**
  - Depende de: T2.7, T2.8, T2.9
  - Checklist: detecção automática do braço; acompanhamento com o violão em movimento lento; PERDIDO → recuperação; FPS ≥ 20 com o SegWorker ativo (anotar o número).
  - Ref.: CA-20, CA-21, CA-22 (parte), RNF-01

---

## Marco 3 — Calibração por pestana + casas projetadas com o violão em movimento

Demonstração: o usuário faz pestana nas casas 1 e 12; trastes, cordas e marcações aparecem e continuam no lugar com o violão em movimento.

- [ ] **T3.1 — `fret_model` e `string_model`**
  - Depende de: T2.1
  - Fazer: f(n), inverso n̂(g), g_dedo(n, β); cordas por margem m, conversões (s, t) ↔ (g, w) ↔ (n̂, k̂) (design §8.1, §8.3).
  - Pronto quando: testes: f(12) = 0,5; ida e volta g ↔ n̂; w_k nos extremos; casos fora da faixa tratados.
  - Ref.: RF-33, RF-37

- [ ] **T3.2 — Solver de calibração 2p, 2p+afunilamento e 3p + validação**
  - Depende de: T3.1
  - Fazer: solver afim, projetivo 1D (linear 3×3 / mínimos quadrados) e com afunilamento; regras de validação (design §8.5); teste sintético com θ de 0 a 40° e ruído de 0 a 6 px.
  - Pronto quando: sem ruído, o 3p recupera os trastes com erro < 0,01 px; tabela de erro × (θ, ruído, modelo) gerada e anotada aqui; validação rejeita os casos de CA-33.
  - Ref.: RF-32, RF-34, RF-35, CA-33, K10

- [ ] **T3.3 — Detector de pestana**
  - Depende de: T3.1, T2.8
  - Fazer: critérios de design §8.4 (indicador estendido, perpendicular, cobertura, sobre o braço, reta 6-7-8 × linha central), janela de estabilidade de 1 s e mediana; registro de motivos de rejeição por quadro (para depuração e avaliação).
  - Pronto quando: em replay com 10 pestanas reais e 10 "não pestanas" (mão relaxada, acorde, movimento), ≥ 9/10 pestanas são capturadas e ≤ 1/10 falso positivo.
  - Ref.: RF-30, RF-31, CA-30, CA-31, K4

- [ ] **T3.4 — Fluxo de calibração (estado CALIBRANDO + UI)**
  - Depende de: T3.2, T3.3
  - Fazer: etapas 1 → 12 (e 5 quando `calib.mode: 3p`), instruções, barra de progresso, sinal visual de captura, validar, confirmar (ENTER) ou refazer (R), timeout com oferta de modo manual.
  - Pronto quando: a calibração ao vivo termina em ≤ 30 s em 4 de 5 tentativas (anotar os tempos).
  - Ref.: RF-30, RF-35, CA-30, CA-37, RNF-10

- [ ] **T3.5 — Calibração manual por cliques**
  - Depende de: T3.2
  - Fazer: modo de clique (nut, traste 12, opcional traste 5) num quadro congelado; mesmo solver.
  - Pronto quando: trastes gerados e desenhados como na pestana.
  - Ref.: RF-36, CA-36

- [ ] **T3.6 — Projeção e overlay de trastes, cordas, marcações e células**
  - Depende de: T3.4
  - Fazer: trastes 0–12, 6 cordas, pontos de marcação (3, 5, 7, 9, 12 duplo), função `cell_center(k, n)` e polígono da célula na imagem via H_t; raio da bolinha proporcional ao tamanho local.
  - Pronto quando: com o violão parado, os trastes desenhados coincidem visualmente com os reais; com movimento lento, continuam alinhados (vídeo de evidência).
  - Ref.: RF-42, CA-32, CA-34

- [ ] **T3.7 — Suavização (One Euro)**
  - Depende de: T3.6
  - Fazer: filtro One Euro nos pontos de contato dos dedos e nos cantos projetados do braço (ou nos parâmetros de H); liga/desliga por configuração.
  - Pronto quando: jitter dos trastes parado ≤ 2 px (desvio-padrão em 10 s) e atraso visual aceitável em movimento; números com e sem filtro anotados aqui.
  - Ref.: RNF-03

- [ ] **T3.8 — Máquina de estados completa do app**
  - Depende de: T3.4, T2.7
  - Fazer: estados e transições de design §10.1 (SEM_CALIBRACAO.{procurando, braço_ok}, CALIBRANDO, RASTREANDO, PERDIDO), tecla C, T_relock com descarte da calibração; estado no HUD e no JSONL.
  - Pronto quando: testes unitários da tabela de transições (eventos simulados); ao vivo, CA-22 completo (sai e volta sem recalibrar).
  - Ref.: RF-26, RF-27, RF-38, CA-22

- [ ] **T3.9 — Inclinação pós-calibração (opcional)**
  - Depende de: T3.6
  - Fazer: ρ/ρ₀ e parâmetro c da calibração no HUD; aviso por índice de compressão (design §7.5).
  - Pronto quando: os valores mudam de forma coerente ao girar o violão (vídeo).
  - Ref.: RF-28

- [ ] **T3.10 — Salvar/carregar calibração (opcional)**
  - Depende de: T3.8
  - Fazer: serializar K (imagem, descritores, bordas) + calibração; ao carregar, relocalizar.
  - Pronto quando: fechar e reabrir o app com a câmera no mesmo lugar recupera a grade sem nova pestana.
  - Ref.: RF-39

- [ ] **T3.11 — Demo do Marco 3**
  - Depende de: T3.6, T3.7, T3.8
  - Checklist: calibração por pestana completa; casas acompanhando o violão (translação ±20 cm, rotação ±30° no plano); perda e recuperação sem recalibrar; FPS ≥ 20.
  - Ref.: CA-30, CA-32, CA-34, CA-22, RNF-01

---

## Marco 4 — Jogo com chart, bolinhas, acerto e pontuação

Demonstração: o usuário escolhe um chart, o metrônomo visual conta os tempos, as bolinhas aparecem nas casas certas, os acertos são julgados e pontuados, e a tela final mostra o resultado.

- [ ] **T4.1 — Formato de chart, validação e exemplos**
  - Depende de: T1.2
  - Fazer: `game/chart.py` (design §9.1), mensagens de erro com índice da nota; script gerador do chart "Pentatônica menor de Lá — casa 5" (design §9.1.1) e os arquivos de 60 e 90 BPM em `charts/`.
  - Pronto quando: testes para chart válido e para cada tipo de erro de CA-50; teste confere as 23 notas da pentatônica (corda, casa e tempo).
  - Ref.: RF-50, RF-59, CA-50

- [ ] **T4.2 — Relógio de jogo e metrônomo visual**
  - Depende de: T4.1
  - Fazer: `GameClock` (`perf_counter`), metrônomo visual (pulso no HUD a cada tempo) e contagem de 4 tempos, pausa/retomada; `t_jogo(frame)` com latência da câmera e offset do usuário.
  - Pronto quando: em replay, cada pulso do metrônomo cai no quadro mais próximo do tempo correspondente; pausa e retomada preservam o tempo (testes unitários).
  - Ref.: RF-51, K11

- [ ] **T4.3 — Agendador e renderização das notas**
  - Depende de: T4.2, T3.6
  - Fazer: notas visíveis em [t − T_lead, t + W_depois]; anel de aproximação (padrão) e modo deslizante; cores por corda; feedback de julgamento.
  - Pronto quando: em replay, o anel se fecha no quadro mais próximo de t_nota (diferença ≤ 1 quadro, teste automatizado sobre o log).
  - Ref.: RF-52, CA-51

- [ ] **T4.4 — Contatos dos dedos → (n̂, k̂) e pertinência a células**
  - Depende de: T3.6
  - Fazer: ponto de contato com γ; conversão para o referencial do braço; pertinência com ε_n e modos zonas/exata; overlay de depuração mostrando (n̂, k̂) de cada dedo.
  - Pronto quando: testes unitários de pertinência (fronteiras, zonas); ao vivo, o dedo indicado por depuração bate com a casa real.
  - Ref.: RF-12, RF-53, CA-54

- [ ] **T4.5 — Experimento de viabilidade de cordas/casas (calibra tolerâncias)**
  - Depende de: T4.4, T1.4
  - Fazer: protocolo de evaluation §5.5 (posicionar o dedo em células conhecidas, a 0,7 e a 1,0 m); distribuições de erro de n̂ e k̂; escolher γ, ε_n e ε_k; decidir se o modo "exata" é viável.
  - Pronto quando: gráficos e tabela de confusão anotados aqui; valores escolhidos gravados em `default.yaml`.
  - Ref.: RF-53, K2

- [ ] **T4.6 — Juiz de acertos (só visão)**
  - Depende de: T4.4
  - Fazer: presença com debounce adaptativo ao FPS; regra perfeito/bom/erro (design §9.4); log de julgamento.
  - Pronto quando: testes com linhas do tempo sintéticas cobrem CA-52, CA-52b, CA-53, notas repetidas e FPS baixo.
  - Ref.: RF-54, CA-52, CA-52b, CA-53

- [ ] **T4.7 — Placar, combo e HUD do jogo**
  - Depende de: T4.6
  - Fazer: `score.py` (design §9.5); HUD com pontos, combo, multiplicador e progresso.
  - Pronto quando: testes de CA-55 e da fórmula de precisão/conceito.
  - Ref.: RF-55, CA-55

- [ ] **T4.8 — Fluxo do jogo: menu, contagem, pausa, fim e resultados**
  - Depende de: T4.3, T4.7, T3.8
  - Fazer: estados de design §10.2; pausa automática em PERDIDO com contagem; tela final; JSON de resultados.
  - Pronto quando: partida completa ao vivo com a pentatônica a 60 e a 90 BPM; CA-56 e CA-57 verificados.
  - Ref.: RF-56, RF-57, RF-58, CA-56, CA-57

- [ ] **T4.9 — Demo do Marco 4**
  - Depende de: T4.8
  - Checklist: menu → partida → tela final, com o violão se movendo um pouco durante a partida; pausa por perda e retomada; FPS ≥ 20 com o jogo ativo; resultado JSON salvo.
  - Ref.: CA-51 a CA-57, CA-70

---

## Marco 5 — Avaliação e comparação com B e C

Demonstração/entrega: tabelas e figuras do relatório, comparando A com B e C (se implementados), tendo a anotação manual como *ground truth*.

- [ ] **T5.1 — Ferramenta de anotação manual**
  - Depende de: T1.4
  - Fazer: `eval/annotate.py`: amostrar quadros (1 por segundo; 2 por segundo nos trechos em movimento); clicar nas interseções do nut e dos trastes 1, 5, 7 e 12 com as duas bordas do braço; marcar intervalos "violão visível / fora do quadro"; papel das mãos; célula pressionada (para E5). Zoom, desfazer e atalhos de teclado; salvar em JSON.
  - Pronto quando: anotar 20 quadros leva ≤ 15 min; reanotar os mesmos quadros dá diferença mediana ≤ 3 px; o arquivo é lido pelo script de métricas.
  - Ref.: RF-72

- [ ] **T5.2 — Gravação do conjunto de avaliação**
  - Depende de: T1.4
  - Fazer: protocolo de evaluation §2 (condições, repetições, roteiro de cada clipe); inventário em `data/eval/index.csv` com as propriedades efetivas da câmera.
  - Pronto quando: todos os clipes da matriz reduzida gravados e conferidos (braço inteiro visível nos trechos previstos do roteiro).
  - Ref.: evaluation §2

- [ ] **T5.3 — Anotação do conjunto**
  - Depende de: T5.1, T5.2
  - Fazer: anotar os quadros amostrados de todos os clipes; anotar 10% duas vezes (variabilidade do anotador).
  - Pronto quando: todos os quadros amostrados anotados; concordância entre as duas anotações reportada.
  - Ref.: RF-72, evaluation §3

- [ ] **T5.4 — Runner offline**
  - Depende de: T5.2, T3.8
  - Fazer: `eval/runner.py`: executa qualquer configuração (localizador, tracker, modelo de calibração, filtro) sobre os clipes; calibração pela pestana do próprio clipe **ou** calibração "oráculo" (nut e traste 12 tirados da anotação do quadro de calibração); saída JSONL por quadro.
  - Pronto quando: A roda em todos os clipes; duas execuções iguais produzem a mesma saída (RNF-08).
  - Ref.: RF-71, RNF-08, CA-03

- [ ] **T5.5 — Scripts de métricas e figuras**
  - Depende de: T5.3, T5.4
  - Fazer: `eval/metrics.py` e `eval/plots.py` com as métricas de evaluation §4 (erro em px e em % da casa, jitter, perda, recuperação, FPS, latência por estágio) e os testes estatísticos.
  - Pronto quando: tabela de A gerada automaticamente a partir dos logs e da anotação; testes unitários das métricas com dados sintéticos.
  - Ref.: RF-73, CA-71

- [ ] **T5.6 — Módulo B: YOLOE/YOLO-World (opcional)**
  - Depende de: T2.3, T2.5
  - Fazer: `yolo_locator.py`, modos B-quadro e B+track (design §7.2); benchmark de latência.
  - Pronto quando: roda pelo runner em todos os clipes.
  - Ref.: RF-29 (B)

- [ ] **T5.7 — Módulo C: Canny + Hough (opcional)**
  - Depende de: T2.1, T3.1
  - Fazer: `classic_locator.py` (design §7.3): eixo pelas cordas, trastes por projeção de gradiente, ajuste 12-TET.
  - Pronto quando: roda pelo runner em todos os clipes.
  - Ref.: RF-29 (C)

- [ ] **T5.8 — Variante A2: SAM 2 video predictor (opcional)**
  - Depende de: T2.4
  - Fazer: tracker por propagação de máscara (design §7.1).
  - Pronto quando: roda pelo runner; latência medida.
  - Ref.: RF-29

- [ ] **T5.9 — Experimentos de localização e rastreamento (E1, E2)**
  - Depende de: T5.5 (+ T5.6, T5.7, T5.8 se feitas)
  - Fazer: evaluation §5.1 e §5.2.
  - Pronto quando: tabelas e figuras de E1 e E2 prontas.
  - Ref.: RNF-03, RNF-04, RNF-05

- [ ] **T5.10 — Experimentos de jogo e latência (E5, E6)**
  - Depende de: T4.8, T5.5
  - Fazer: evaluation §5.5, §5.6 e §6 (taxa de acerto das notas, confusão de cordas, latência fim-a-fim).
  - Pronto quando: tabelas prontas; `camera.latency_ms` calibrado em `default.yaml`.
  - Ref.: RF-54, RNF-02, K11

- [ ] **T5.11 — Experimentos de calibração (E3, E4)**
  - Depende de: T5.5
  - Fazer: evaluation §5.3 e §5.4 (2p × 2p+afunilamento × 3p; β; taxa de sucesso da pestana).
  - Pronto quando: tabelas prontas; β padrão revisado em `default.yaml` se necessário.
  - Ref.: RF-32, RF-34, RNF-10

- [ ] **T5.12 — Consolidação para o relatório**
  - Depende de: T5.9, T5.10, T5.11
  - Fazer: script único que regenera todas as tabelas/figuras; seção de limitações e ameaças à validade.
  - Pronto quando: `python -m ghirl.eval.report` gera `reports/` do zero a partir dos logs.
  - Ref.: RF-73

---

## Grafo de dependências (resumo)

```
T1.1 ─ T1.2 ─ T1.3 ─ T1.4 ─────────────────────────────────────────┐
  │      │      └── T1.5 ─ T1.6                                    │
  │      │            └── T1.7 ─────────┐                          │
  │      ├─ T2.1 ─ T2.3 ─ T2.4 ─ T2.5 ─ T2.6 ─ T2.7                │
  └─ T2.2 ──┘                     ├─ T2.8   └─ T2.9                │
         │                                                         │
         ├─ T3.1 ─ T3.2 ─ T3.4 ─ T3.6 ─ T3.7                       │
         │    └── T3.3 ─┘   └ T3.5   └─ T3.8 (+T2.7)               │
         └─ T4.1 ─ T4.2 ─ T4.3 (+T3.6) ─ T4.8 ◄ T4.7 ◄ T4.6 ◄ T4.4 (+T3.6)
                                           └─ T4.9 (demo)          │
T5.1 (T1.4) ─┐                                                     │
T5.2 (T1.4) ─┴─ T5.3 ─┐                                            │
T5.2 + T3.8 ─ T5.4 ───┴─ T5.5 ─ T5.9 / T5.10 / T5.11 ─ T5.12
```
