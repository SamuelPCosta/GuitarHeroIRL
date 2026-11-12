# Guitar Hero IRL — Tarefas

> v0.3 · 2026-11-11 · [requisitos](requirements.md) · [design](design.md) · [avaliação](evaluation.md)
> Nada é commitado sem revisão do usuário.

## Feito

- [x] Ambiente (Python 3.12, torch cu128, MediaPipe, ultralytics, PySide6); webcam 720p MJPG a 30 FPS
      com exposição fixa; gravação e replay de sessões.
- [x] Mãos (MediaPipe) e máscara das mãos para o rastreador.
- [x] Achar o braço (só YOLOE → geometria), assíncrono e com aquecimento. SAM removido (medido: não ajudava).
- [x] Seguir o braço (LK + homografia, movimento rápido, relocalização). Gravação de 30 s: 0 quadros perdidos.
- [x] Calibração por pestana confirmada por botão/espaço; rejeita pestanas trocadas e repetidas.
- [x] Interface Qt; só a nota atual sobre o braço; exercícios em JSON.
- [x] Código e testes enxutos (12 módulos, 10 testes).
- [x] Gestos: fechar-abrir = avançar; punho 3 s = recalibrar.
- [x] Ajuste da escala pelos trastes da imagem (vídeo real: erro ~18 px → ~6 px).
- [x] Opção "trastes e notas" (baixa opacidade).
- [x] A nota só vale com o dedilhado da outra mão (qualquer dedo, corda certa, para baixo).
- [x] Violão que sai do quadro é reencontrado sem perder a calibração (sem limite de tempo).
- [x] Recalibrar sozinho pelos trastes quando o quadro-chave muda (gravação: 13 trastes, erro 0,1 px).
- [x] Nota suavizada (filtro One Euro): tremedeira 1,74 → 0,50 px, atraso 2,7 px.
- [x] Mais exercícios: pentatônica forma 2, blues de Lá, Dó maior, Ode à Alegria (6 no total).
- [x] `GuitarHeroIRL.pyw` para abrir com dois cliques.

## A fazer

- [ ] **Teste ao vivo**: calibração por gesto, acerto com dedilhado, ajuste pelos trastes. Ajustar
      limiares (punho/mão aberta, dedilhado, folgas).
- [ ] Avaliação para o relatório ([evaluation.md](evaluation.md)).
