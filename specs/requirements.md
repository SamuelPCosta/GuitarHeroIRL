# Guitar Hero IRL — Requisitos

> Versão 0.2 (revisada: só imagens, sem ArUco, violão de aço, calibração de 2 pontos) · 2026-11-11
> Documentos relacionados: [design.md](design.md) · [tasks.md](tasks.md) · [evaluation.md](evaluation.md)

## 1. Visão geral

Protótipo de visão computacional em Python que, a partir de uma webcam fixa voltada para o usuário, localiza o braço de um violão em tempo real, projeta sobre o vídeo exibido na tela marcações (bolinhas) nas células corda × casa e executa um jogo rítmico: notas de um *chart* aparecem sobre o braço sincronizadas com a música, e o sistema pontua quando a mão que faz as notas ocupa a célula correta na janela de tempo correta.

Objetivo acadêmico: comparar abordagens *zero-shot*/clássicas de localização do braço (SAM + tracking, YOLO open-vocabulary, OpenCV clássico) entre si, usando anotação manual como *ground truth*, com métricas quantitativas. Nesta versão o sistema usa **apenas imagens** (sem áudio).

## 2. Glossário

| Termo | Significado neste projeto |
|---|---|
| **Nut** | Peça na cabeça do braço onde as cordas apoiam (em português também chamada de "pestana" ou "capotraste fixo"). Para evitar ambiguidade, usamos **nut**. |
| **Pestana** | Gesto em que o indicador pressiona todas as cordas em uma mesma casa (*barre*). |
| **Traste** | Barra metálica transversal. O traste *n* delimita o fim da casa *n*. |
| **Casa *n*** | Região entre os trastes *n−1* e *n* (traste 0 = nut). Casa 0 = corda solta. |
| **Corda *k*** | Numeração padrão: 1 = Mi aguda (mais fina), 6 = Mi grave (mais grossa). |
| **Escala (L)** | Distância nut → rastilho (≈ 2 × distância nut → traste 12). No violão do projeto, ≈ 648 mm (a medir). |
| **Referencial do braço** | Sistema de coordenadas preso ao braço: *u* ao longo do braço (a partir do nut) e *w* através do braço (entre as bordas). Ver design §5. |
| **Quadro-chave (K)** | Quadro de referência no qual a geometria do braço foi estimada. O rastreamento fornece a transformação K → quadro atual. |
| **Mão de notas** | Mão esquerda do usuário destro, que pressiona as cordas no braço. |
| **Mão de ritmo** | Mão direita, que toca as cordas perto da boca do violão. |
| **Chart** | Arquivo com a sequência de notas (tempo, corda, casa) de uma música. |
| **Célula** | Região corda *k* × casa *n* no braço. |

## 3. Contexto, atores e premissas

- **Ator único**: o usuário (destro), sentado de frente para a webcam, tocando violão.
- **Hardware**: PC Windows 10, GPU RTX 3060 12 GB, webcam USB ou de notebook (tipicamente 720p a 30 FPS).
- **Software**: Python 3.12, OpenCV, MediaPipe (Hand Landmarker), PyTorch com CUDA, SAM 2/SAM 3, Ultralytics (YOLO-World/YOLOE).
- **Premissas** (os riscos estão em §9):
  - P1. Câmera fixa durante uma sessão; o violão se move.
  - P2. Um só violão e um só usuário na cena.
  - P3. Afinação padrão (E A D G B E). Casas jogáveis de 1 a 12.
  - P4. A mão de notas fica pelo menos parcialmente visível (dedos sobre o espelho do braço, voltados para a câmera).
  - P5. Não há treinamento nem *fine-tuning* de nenhum modelo para o violão (restrição do enunciado).
  - P6. Violão de aço Tagima Dallas (folk): escala ≈ 648 mm, nut ≈ 43 mm, junção com o corpo na casa 14 (valores a medir com régua). Só entram em verificações de plausibilidade, porque a calibração estima a escala em pixels.
  - P7. Nada é colado ou preso no violão (sem marcadores).

## 4. Requisitos funcionais

Prioridade: **M** = obrigatório (*must*), **S** = desejável (*should*), **O** = opcional (*could*).

### 4.1 Captura e reprodução (RF-0x)

| ID | Requisito | Prior. |
|---|---|---|
| RF-01 | Capturar quadros da webcam em uma thread dedicada, com carimbo de tempo monotônico (`perf_counter`) por quadro, descartando quadros antigos (sempre processar o mais recente). | M |
| RF-02 | Permitir configurar resolução, FPS, formato (MJPG) e, quando o driver suportar, desligar auto-exposição/auto-ganho/auto-white-balance e fixar valores. Registrar o FPS **efetivamente** obtido. | M |
| RF-03 | Gravar sessões (vídeo bruto + carimbos de tempo) e reproduzi-las como se fossem a câmera ("modo replay"), de forma determinística, para testes e avaliação. | M |
| RF-04 | Exibir o vídeo com overlay em janela; opção de exibir espelhado (efeito "espelho") **sem** espelhar a imagem processada. | M |

### 4.2 Mãos (RF-1x)

| ID | Requisito | Prior. |
|---|---|---|
| RF-10 | Detectar até 2 mãos por quadro com MediaPipe Hand Landmarker (21 pontos) em modo de vídeo. | M |
| RF-11 | Atribuir o **papel** de cada mão (notas / ritmo) por regra geométrica relativa ao braço (mão mais próxima do braço, do lado do nut = notas), usando o rótulo de lateralidade do MediaPipe apenas como desempate, com mapeamento de rótulo configurável conforme a imagem esteja ou não espelhada. | M |
| RF-12 | Expor, para a mão de notas, as pontas dos dedos indicador, médio, anelar e mínimo (pontos 8, 12, 16, 20) e um "ponto de contato" corrigido (ver design §8.3) em coordenadas de imagem e no referencial do braço. | M |
| RF-13 | Gerar uma máscara (casco convexo dilatado) das mãos para excluir pixels de mão do rastreamento do braço. | M |

### 4.3 Localização e rastreamento do braço (RF-2x)

| ID | Requisito | Prior. |
|---|---|---|
| RF-20 | Localizar o braço no quadro inicial com SAM (prompt de texto em SAM 3, ou clique/caixa em SAM 2) — **módulo A**. | M |
| RF-21 | Isolar o braço do restante do violão (corpo, cabeça) quando a máscara cobrir o violão inteiro, pelo perfil de largura ao longo do eixo principal. | M |
| RF-22 | Estimar o eixo do braço (ponto + direção), as duas bordas laterais e a largura ao longo do eixo (PCA/`minAreaRect` + ajuste de retas às bordas). | M |
| RF-23 | Rastrear o braço quadro a quadro com método leve (Lucas-Kanade sobre pontos de textura do braço/cabeça), estimando uma homografia K → quadro atual com RANSAC, excluindo pixels das mãos (RF-13). | M |
| RF-24 | Calcular um indicador de qualidade do rastreamento por quadro (nº de *inliers*, erro de reprojeção, variação de escala/forma). | M |
| RF-25 | Re-segmentar periodicamente (assíncrono, sem bloquear o laço principal) e quando a qualidade cair, corrigindo deriva. | M |
| RF-26 | Declarar o estado **PERDIDO** quando a qualidade ficar abaixo do limiar por mais de *N* quadros. | M |
| RF-27 | Tentar relocalizar automaticamente (casamento de features ORB contra o quadro-chave e/ou re-segmentação) e, se der certo, voltar a **RASTREANDO** mantendo a calibração. | M |
| RF-28 | Estimar a inclinação do braço (rotação no plano e fora do plano) e exibir um aviso de "posição ruim" quando ultrapassar os limites recomendados (ver design §7.5). | S |
| RF-29 | Oferecer implementações intercambiáveis de localização sob a mesma interface: **A** (SAM + tracking), **B** (YOLO-World/YOLOE "guitar" + refino), **C** (Canny + Hough). Selecionável por configuração. | A: M · B, C: O |

### 4.4 Calibração (RF-3x)

| ID | Requisito | Prior. |
|---|---|---|
| RF-30 | Guiar o usuário na calibração por pestana: casa 1 e depois casa 12, segurando ~1 s em cada, com instruções e barra de progresso na tela. | M |
| RF-31 | Detectar a pestana: mão de notas sobre o braço, indicador estendido (aproximadamente reto), aproximadamente perpendicular ao eixo, cobrindo boa parte da largura do braço, e estável durante a janela de captura. | M |
| RF-32 | Resolver a posição do nut e a escala L no referencial do braço a partir das duas medidas, considerando o deslocamento do dedo em relação ao traste (parâmetro β). | M |
| RF-33 | Gerar as posições de todos os trastes 0–12 pela regra de 12-TET: *d(n) = L·(1 − 2^(−n/12))*. | M |
| RF-34 | Modo de calibração opcional com 3 pontos (casas 1, 12 e 5), ativado por configuração, que modela a perspectiva ao longo do eixo (homografia 1D). O padrão é 2 pontos (RF-32). | O |
| RF-35 | Validar a calibração (sinal, faixa plausível de L em relação à largura do braço, resíduo no 3º ponto quando houver) e pedir confirmação ou refazer. | M |
| RF-36 | Calibração manual por cliques (nut e traste 12) como alternativa e para avaliação. | M |
| RF-37 | Modelar as 6 cordas através da largura do braço (frações configuráveis entre as bordas, interpoladas ao longo do eixo). | M |
| RF-38 | Re-calibração: tecla dedicada a qualquer momento; obrigatória se o braço ficar PERDIDO por mais de *T_relock* sem relocalização. | M |
| RF-39 | Salvar e carregar a calibração (com o quadro-chave) para reuso na mesma sessão de câmera. | O |

### 4.5 Overlay (RF-4x)

| ID | Requisito | Prior. |
|---|---|---|
| RF-40 | Desenhar overlay sobre o vídeo: no Marco 1, uma grade fixa de bolinhas; depois, a geometria real do braço. | M |
| RF-41 | Modo de depuração: eixo, bordas, pontos rastreados, máscara, mãos, estado e qualidade. | M |
| RF-42 | Desenhar trastes, cordas, marcações (casas 3, 5, 7, 9, 12) e bolinhas em células arbitrárias, sempre calculadas no referencial do braço e projetadas no quadro atual. | M |
| RF-43 | HUD com FPS, latência por módulo e estado atual. | M |

### 4.6 Jogo (RF-5x)

| ID | Requisito | Prior. |
|---|---|---|
| RF-50 | Ler charts em JSON (metadados + lista de notas com tempo, corda, casa e duração opcional) e validar o esquema. | M |
| RF-51 | Relógio de jogo único (monotônico), com metrônomo **visual** (pulso no HUD a cada tempo e contagem inicial) e *offset* de latência da câmera configurável. | M |
| RF-52 | Exibir as notas futuras na célula correspondente com antecedência configurável (anel de aproximação que se fecha no tempo da nota; modo alternativo: bolinha deslizando pela corda até a casa). | M |
| RF-53 | Detectar presença da mão de notas na célula correta, com tolerâncias configuráveis ao longo e através do braço, incluindo um modo "zonas de cordas" (graves/médias/agudas) e um modo "corda exata". | M |
| RF-54 | Julgar cada nota como **perfeito**, **bom** ou **erro**, com janelas de tempo configuráveis e *debounce* (mínimo de quadros consecutivos). | M |
| RF-55 | Pontuação, combo, multiplicador por combo e HUD do jogo. | M |
| RF-56 | Pausar o jogo automaticamente quando o braço estiver PERDIDO e retomar com contagem regressiva. | S |
| RF-57 | Tela final: pontuação, % de acerto, contagem por julgamento, combo máximo, conceito (S/A/B/C/D). | M |
| RF-58 | Salvar o resultado de cada partida e o log de julgamentos (JSON) para avaliação. | M |
| RF-59 | Incluir o chart "Pentatônica menor de Lá — casa 5" (sobe e desce; design §9.1) em dois andamentos (60 e 90 BPM). | M |

### 4.7 Instrumentação e avaliação (RF-7x)

| ID | Requisito | Prior. |
|---|---|---|
| RF-70 | Medir o tempo de cada estágio do pipeline por quadro e registrar em JSONL (quadro, timestamps, estado, pose, mãos, julgamentos). | M |
| RF-71 | Executar qualquer configuração de pipeline *offline* sobre vídeos gravados e exportar resultados por quadro. | M |
| RF-72 | Gerar *ground truth* por anotação manual assistida (ferramenta de clique em quadros amostrados). | M |
| RF-73 | Scripts que calculam as métricas de [evaluation.md](evaluation.md) e geram tabelas/figuras. | M |

### 4.8 Configuração (RF-8x)

| ID | Requisito | Prior. |
|---|---|---|
| RF-80 | Toda constante ajustável (resoluções, limiares, tolerâncias, janelas, β, escala, frações das cordas, módulo de localização) em arquivo YAML com valores padrão documentados; sobrescrita por linha de comando. | M |
| RF-81 | Suporte a canhoto por configuração (espelha os papéis das mãos e a orientação do braço). Implementado, mas não testado. | O |

## 5. Requisitos não funcionais

| ID | Requisito | Meta / medição |
|---|---|---|
| RNF-01 | **Desempenho**: pipeline completo (captura → mãos → braço → jogo → render) a ≥ 20 FPS sustentados em RASTREANDO, na máquina-alvo, a 1280×720 (ou 960×540 se necessário). | Média e P5 do FPS em 60 s de uso; medido pelo profiler (RF-70). |
| RNF-02 | **Latência**: latência de processamento (captura do quadro → overlay exibido) ≤ 80 ms em média; latência fim-a-fim (*glass-to-glass*) medida e reportada. | Profiler + teste com cronômetro filmado (evaluation §6). |
| RNF-03 | **Estabilidade**: *jitter* da posição projetada dos trastes com violão parado ≤ 2 px (desvio-padrão) a 720p. | evaluation §4.2. |
| RNF-04 | **Precisão**: erro mediano de posição dos trastes 1–12 ≤ 25% do espaçamento local da casa contra a anotação manual, na condição base. | evaluation §4.1. |
| RNF-05 | **Robustez**: com movimento lento do violão (translação ≤ 20 cm/s, rotação ≤ 20°/s), taxa de quadros em PERDIDO ≤ 5% na condição base. | evaluation §4.3. |
| RNF-06 | **Não bloqueio**: modelos pesados (SAM, YOLO) executam de forma assíncrona; o laço principal nunca espera por eles depois da inicialização. | Revisão + profiler: nenhum quadro do laço > 100 ms por causa de segmentação. |
| RNF-07 | **Modularidade**: localizadores (A–D), rastreadores e detectores de acerto trocáveis por configuração, sem alterar o laço principal. | Revisão de interfaces (design §4). |
| RNF-08 | **Reprodutibilidade**: o modo replay produz resultados idênticos para a mesma entrada e configuração (exceto timings). Sementes fixas onde houver aleatoriedade (RANSAC). | Teste de integração compara duas execuções. |
| RNF-09 | **Portabilidade**: roda no Windows 10 com Python 3.12; instalação documentada em um README com uma lista de dependências fixada. | Instalação limpa segue o README. |
| RNF-10 | **Usabilidade**: calibração concluída em ≤ 30 s por um usuário que segue as instruções da tela. | evaluation §4.4. |
| RNF-11 | **Testabilidade**: a matemática (calibração, geometria, julgamento, pontuação) é coberta por testes unitários (pytest) com dados sintéticos. | `pytest` passa; cobertura desses módulos ≥ 80%. |

## 6. Critérios de aceitação (Dado / Quando / Então)

### Captura e overlay
- **CA-01** (RF-01, RF-43) — *Dado* a webcam conectada, *quando* inicio o app em modo câmera, *então* a janela mostra o vídeo ao vivo e o HUD exibe o FPS de captura e o de processamento.
- **CA-02** (RF-02) — *Dado* uma configuração com exposição manual, *quando* o driver aceita o ajuste, *então* o log registra os valores aplicados; *quando* não aceita, *então* o log emite aviso e o app continua.
- **CA-03** (RF-03, RNF-08) — *Dado* uma sessão gravada, *quando* executo o app duas vezes em modo replay com a mesma configuração, *então* os logs de pose e julgamento são idênticos.
- **CA-04** (RF-04) — *Dado* o modo espelho ativado, *quando* levanto a mão direita, *então* ela aparece do lado direito da tela, e o papel atribuído às mãos não muda em relação ao modo sem espelho.

### Mãos
- **CA-10** (RF-10, RF-11) — *Dado* o usuário destro tocando com o braço visível, *quando* as duas mãos estão em cena, *então* a mão sobre o braço é rotulada "notas" em ≥ 95% dos quadros em que ambas são detectadas (anotação manual de 300 quadros), com ou sem espelhamento.
- **CA-11** (RF-13) — *Dado* uma mão sobre o braço, *quando* o rastreador seleciona pontos, *então* nenhum ponto rastreado cai dentro da máscara da mão (verificável no modo de depuração).

### Braço
- **CA-20** (RF-20–22) — *Dado* o violão na posição recomendada, *quando* o app inicia o módulo A, *então* em ≤ 5 s o eixo e as bordas do braço aparecem no modo de depuração, com o eixo a ≤ 5° do eixo anotado manualmente.
- **CA-21** (RF-23, RNF-01) — *Dado* o braço localizado, *quando* movo o violão lentamente, *então* o overlay do eixo acompanha o braço e o FPS fica ≥ 20.
- **CA-22** (RF-26, RF-27) — *Dado* RASTREANDO, *quando* tiro o violão de cena por 2 s e o recoloco em posição parecida, *então* o estado passa para PERDIDO, e em ≤ 3 s volta para RASTREANDO sem nova calibração, com erro de traste dentro de RNF-04.
- **CA-23** (RF-28) — *Dado* o braço rodado mais de 45° fora do plano (perfil), *quando* a inclinação estimada passa do limite, *então* a tela mostra o aviso de posição ruim.
- **CA-24** (RF-29, RNF-07) — *Dado* `neck.locator: yolo` (ou `classic`) na configuração, *quando* inicio o app, *então* o pipeline roda com esse localizador sem mudança de código.

### Calibração
- **CA-30** (RF-30, RF-31) — *Dado* SEM_CALIBRACAO, *quando* faço pestana na casa 1 e seguro por 1 s, *então* o sistema confirma a captura com um sinal visual e pede a casa 12.
- **CA-31** (RF-31) — *Dado* CALIBRANDO, *quando* apenas encosto a mão no braço sem pestana (dedos dobrados) ou movo a mão, *então* nenhuma captura é feita.
- **CA-32** (RF-32, RF-33, RNF-04) — *Dado* as duas capturas válidas, *quando* a calibração termina, *então* os trastes 1–12 são desenhados e o erro mediano contra a anotação manual é ≤ 25% do espaçamento local.
- **CA-33** (RF-35) — *Dado* medidas inconsistentes (por exemplo, casa 12 capturada mais perto do nut que a casa 1), *quando* o solver roda, *então* a calibração é rejeitada com mensagem e o usuário recomeça.
- **CA-34** (RF-42, requisito central) — *Dado* calibrado, *quando* translado/giro o violão lentamente no plano da imagem (±20 cm, ±30°), *então* as casas desenhadas continuam sobre as casas reais (mesmo critério de RNF-04).
- **CA-35** (RF-34) — *Dado* o modo 3 pontos com o braço girado ~30° fora do plano, *quando* calibro, *então* o erro mediano é menor ou igual ao do modo 2 pontos na mesma gravação.
- **CA-36** (RF-36) — *Dado* o modo manual, *quando* clico no nut e no traste 12, *então* a grade de trastes é gerada como na calibração por pestana.
- **CA-37** (RNF-10) — *Dado* um usuário novo seguindo as instruções, *quando* inicia a calibração, *então* ela termina em ≤ 30 s em ≥ 4 de 5 tentativas.

### Jogo
- **CA-50** (RF-50) — *Dado* um chart com campo inválido (corda 7, tempo negativo, notas fora de ordem), *quando* o carrego, *então* o app informa o erro com a posição da nota e não inicia a partida.
- **CA-51** (RF-52) — *Dado* uma partida em andamento, *quando* faltam *T_lead* segundos para uma nota, *então* seu anel aparece na célula correta e se fecha exatamente no tempo da nota (diferença ≤ 1 quadro em replay).
- **CA-52** (RF-53, RF-54) — *Dado* uma nota (corda 3, casa 5) em t = 10,0 s, *quando* um dedo da mão de notas está na célula (com *debounce* de ≥ 2 quadros) no quadro mais próximo de t, *então* a nota é julgada **perfeito**.
- **CA-52b** (RF-54) — *Dado* a mesma nota, *quando* o dedo só chega à célula entre t e t + 0,2 s (ou esteve lá entre t − 0,6 s e t, mas saiu antes de t), *então* a nota é julgada **bom**.
- **CA-53** (RF-54) — *Dado* a mesma nota, *quando* nenhum dedo fica na célula durante a janela [t − 0,6 s, t + 0,2 s], *então* a nota é julgada **erro** e o combo zera.
- **CA-54** (RF-53) — *Dado* o modo "zonas", *quando* o dedo está na casa correta mas uma corda adjacente dentro da mesma zona, *então* conta como acerto; *dado* o modo "corda exata", *então* conta como erro.
- **CA-55** (RF-55) — *Dado* 10 acertos seguidos, *quando* acerto a 11ª nota, *então* o multiplicador passa para ×2 (regra no design §9.5).
- **CA-56** (RF-56) — *Dado* uma partida, *quando* o braço fica PERDIDO, *então* o jogo pausa e, ao voltar a RASTREANDO, há contagem de 3 s antes de retomar.
- **CA-57** (RF-57, RF-58) — *Dado* o fim do chart, *quando* a última nota é julgada, *então* a tela final aparece e um JSON de resultados é salvo.

### Desempenho e avaliação
- **CA-70** (RNF-01, RF-70) — *Dado* RASTREANDO com jogo ativo, *quando* rodo 60 s, *então* o relatório do profiler mostra FPS médio ≥ 20 e tempo por estágio.
- **CA-71** (RF-71, RF-73) — *Dado* o conjunto de vídeos de avaliação, *quando* rodo o script de avaliação para A (e B/C, se implementados), *então* são geradas as tabelas de erro, jitter, perda, FPS e latência.

## 7. Fora de escopo

- Projeção física sobre o violão (projetor). Só overlay na tela.
- Treinar ou fazer *fine-tuning* de qualquer detector/segmentador para o violão ou trastes.
- Reconhecimento de acordes (várias notas simultâneas) e transcrição polifônica de áudio.
- Detecção de técnicas (bend, slide, hammer-on, vibrato) e de pressão real da corda (a visão só mede presença).
- Casas acima da 12 e cordas soltas (casa 0), que a visão não consegue confirmar.
- **Áudio** (microfone, pitch, onsets, música de fundo, metrônomo sonoro): adiado. Esta versão usa só imagens.
- Marcadores fiduciais (ArUco ou similares) ou qualquer coisa colada no violão.
- Geração automática de charts a partir de áudio. Editor gráfico de charts.
- Múltiplos usuários/violões, guitarras elétricas de formatos exóticos, câmera móvel, celular.
- Suporte a outros sistemas operacionais além do Windows (pode funcionar, mas não é testado).
- Canhotos: apenas configurável (RF-81), sem validação.

## 8. Restrições

- R1. Somente modelos prontos (zero-shot/open-vocabulary) e OpenCV clássico para o violão. MediaPipe Hands para as mãos.
- R2. Toda geometria do jogo é definida no referencial do braço; coordenadas de imagem são só para exibição e medição.
- R3. Entregas incrementais: cada marco de [tasks.md](tasks.md) pode ser demonstrado sozinho.
- R4. Implementação começa só depois da aprovação desta spec, uma task por vez.

## 9. Riscos

| # | Risco | Prob. | Impacto | Mitigação |
|---|---|---|---|---|
| K1 | **Braço visto de lado/perfil**: com câmera frontal, se o tampo gira (yaw) ou inclina para trás (tilt), as casas ficam comprimidas e as cordas se juntam. | Alta | Alto | Posição recomendada (design §7.5) mostrada na tela; estimativa de inclinação e aviso (RF-28); calibração opcional de 3 pontos (RF-34). |
| K2 | **Resolução insuficiente para separar cordas**: a 1–1,2 m e 720p, as cordas ficam a ~5–7 px de distância; o erro da ponta do dedo no MediaPipe é da mesma ordem. | Alta | Alto | Modo "zonas de cordas" como padrão; câmera mais próxima (0,6–0,9 m) enquadrando o braço; experimento de viabilidade (T4.5). |
| K3 | **Oclusão do braço pela mão de notas** e pelo braço direito/corpo. | Alta | Médio | Máscara das mãos excluída do tracking (RF-13); pontos na cabeça do violão e na junção com o corpo; homografia robusta (RANSAC); relocalização (RF-27). |
| K4 | **MediaPipe falha na pestana** (dedos dobrados sobre um objeto, mão parcialmente atrás do braço). | Média | Alto | Critérios tolerantes, uso dos pontos 6–8 do indicador (não só a ponta), mediana em 1 s; calibração manual por clique (RF-36). |
| K5 | **Lateralidade invertida** com imagem espelhada ou erro do classificador. | Alta | Médio | Papel atribuído por geometria (RF-11); teste empírico na T1.7. |
| K6 | **Webcam com auto-exposição e baixo FPS** (cai para 15 FPS em pouca luz; *motion blur*). | Alta | Alto | Fixar exposição/ganho (RF-02); recomendar iluminação frontal; MJPG; medir FPS real; ajustar janelas de acerto ao FPS medido. |
| K7 | **SAM lento na RTX 3060** (SAM 3 ~ centenas de ms por imagem). | Alta | Médio | Execução assíncrona só em quadros-chave (RNF-06); SAM 2.1 tiny/small como alternativa; YOLOE-seg como gerador de prompt. |
| K8 | **SAM 3 indisponível ou difícil de instalar no Windows** (pesos com acesso controlado, dependências). | Média | Médio | Alternativa SAM 2.1 com clique do usuário ou caixa do YOLO-World; verificar na T2.2 antes de tudo. |
| K9 | **Deriva do tracking** ao longo de minutos. | Média | Médio | Re-segmentação periódica e verificação de IoU (RF-25); re-ancoragem ao quadro-chave por ORB. |
| K10 | **Perspectiva ao longo do braço** torna o modelo afim de 2 pontos impreciso. | Média | Médio | Modelo projetivo 1D com 3 pontos; quantificar o erro em evaluation §5.3. |
| K11 | **Latência da câmera** (~50–150 ms) atrasa o que o sistema vê em relação ao relógio do jogo. | Alta | Médio | Offset de latência configurável e medido (evaluation §6); janelas assimétricas. |
| K13 | **Prazo desconhecido**. | — | Alto | Marcos demonstráveis; B e C marcados como opcionais. |
| K14 | **Compatibilidade de versões** (MediaPipe × Python × PyTorch × CUDA no Windows). | Média | Médio | Fixar versões na T1.1; ambientes separados se necessário (por exemplo, um processo de segmentação). |
