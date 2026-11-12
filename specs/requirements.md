# Guitar Hero IRL — Requisitos

> v0.3 · 2026-11-11 · [design](design.md) · [tarefas](tasks.md) · [avaliação](evaluation.md)

## O que é

Protótipo de visão computacional (disciplina de mestrado). Uma webcam fixa filma o usuário tocando um
violão de aço (Tagima Dallas, destro). O app acha o braço sozinho, segue o violão em tempo real e mostra
**uma nota por vez** de um exercício sobre o braço. A nota vale quando um dedo está na casa/corda certa
**e** um dedo da outra mão toca essa corda para baixo (dedilhado); então aparece a próxima.

## Restrições

- Sem treinar nem ajustar modelos: só modelos prontos (YOLOE para o violão, MediaPipe Hands para as mãos) e OpenCV clássico.
- Sem áudio. Sem marcadores no violão (nada de ArUco).
- Toda a geometria vive no referencial do braço; a imagem só serve para mostrar e medir.
- ≥ 20 FPS na RTX 3060 com webcam 720p. Windows 10, Python 3.12.
- Protótipo: o mínimo de código, pastas, testes e documentação que funcione.

## Fluxo do usuário

1. Abre o app com dois cliques (`GuitarHeroIRL.pyw`). Fica de frente para a câmera com o braço inteiro
   (inclusive a cabeça) no quadro.
2. O app acha o violão (contorno verde).
3. **Calibração**: pestana na casa 1 → *avançar*; pestana na casa 12 → *avançar*. O app mostra a etapa e
   se a pestana foi reconhecida (indicador verde) ou o motivo da recusa.
4. **Exercício**: a nota atual aparece sobre o braço (bolinha com a cor da corda). Dedo na nota +
   corda tocada com a outra mão = acerto → anel verde → próxima nota. No fim, *avançar* recomeça.

## Controles

A câmera fica longe do teclado, então tudo funciona por gesto, com qualquer mão:

| Ação | Gesto | Alternativa |
|---|---|---|
| Avançar (capturar pestana; recomeçar exercício no fim) | fechar e abrir a mão | botão, espaço |
| Recomeçar a calibração | mão fechada por 3 s (anel de progresso na tela) | botão |

No jogo, gestos de uma mão sobre o violão são ignorados (dedilhando, a mão fecha às vezes).

Se o violão sair do quadro (ou for coberto), ao voltar o app o reencontra sozinho e **mantém a calibração**.
Se ele voltar numa posição que não dá para casar com a referência, a calibração é reajustada sozinha
pelos trastes; só se isso falhar o app pede as pestanas de novo.

## Interface (Qt)

Vídeo + painel: estado, mensagens, etapa da calibração, exercício (lista dos JSON), nota atual, progresso,
tempo. Opções: espelhar vídeo; **trastes e notas** (T) — todos os trastes e todas as notas do exercício
em baixa opacidade; **depuração** (D) — pontos rastreados, esqueleto das mãos, trastes achados.

## Exercícios

Arquivos JSON em `exercises/` (formato no [design](design.md#exercícios)). Inclusos: pentatônica de Lá
menor (formas 1 e 2), blues de Lá, escala maior de Dó (casa 8), cromático 1-2-3-4 e Ode à
Alegria (Beethoven, domínio público).

## Fora de escopo

Áudio; acordes; cordas soltas e casas acima da 12; técnicas (bend, slide); canhotos; outros SOs.

## Riscos

- Violão escuro sobre roupa escura: máscara parcial do braço.
- Perspectiva forte (braço inclinado em relação à câmera): trastes comprimidos.
- Cordas a poucos px umas das outras na imagem.
- Dedilhado em 30 FPS é rápido e pequeno na imagem.
- Gestos confundidos com a mão dedilhando.
