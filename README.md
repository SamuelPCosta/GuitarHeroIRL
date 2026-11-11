# Guitar Hero IRL

Protótipo de visão computacional: a webcam filma o violão, o programa desenha as casas e cordas sobre o vídeo em tempo real e pontua quando a nota certa é tocada (só imagens, sem áudio).

Especificação: [specs/](specs/) — [requisitos](specs/requirements.md), [design](specs/design.md), [tarefas](specs/tasks.md), [avaliação](specs/evaluation.md).

## Instalação (Windows 10/11, GPU NVIDIA)

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e driver NVIDIA recente (CUDA 12.8+).

```powershell
uv venv --python 3.12 .venv
# 1) PyTorch com CUDA PRIMEIRO (senão o ultralytics puxa a versão de CPU)
uv pip install --python .venv\Scripts\python.exe -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu128
# 2) resto, com o override que bloqueia opencv-python
uv pip install --python .venv\Scripts\python.exe -r requirements.txt --override overrides.txt
uv pip install --python .venv\Scripts\python.exe -e . --no-deps
```

> Não instale `opencv-python`: o MediaPipe já traz `opencv-contrib-python`, e os dois juntos quebram o módulo `cv2`. O `overrides.txt` impede que o `ultralytics` o puxe.

Modelos (pasta `models/`, fora do git):

```powershell
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

`yoloe-26s-seg-pf.pt` e `sam2.1_t.pt` são baixados automaticamente pelo `ultralytics` na primeira execução. O SAM 3 (`models/sam3.pt`) é opcional e exige pedir acesso em huggingface.co/facebook/sam3.

Verificar o ambiente:

```powershell
.venv\Scripts\python -m ghirl.utils.check_env
.venv\Scripts\python -m pytest
.venv\Scripts\ruff check .
```

`check_env` deve terminar com `OK` e mostrar a GPU.

## Estrutura

```
ghirl/      código (app, capture, hands, neck, calib, game, render, eval, utils)
configs/    configuração YAML
charts/     charts do jogo (JSON)
tests/      testes (pytest)
specs/      especificação
data/       gravações e dados locais (fora do git)
```
