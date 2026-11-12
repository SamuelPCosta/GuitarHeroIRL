# Guitar Hero IRL

A webcam filma você tocando violão; o programa acha o braço, segue o violão e mostra uma nota por vez
sobre ele. A nota vale quando o dedo está nela e um dedo da outra mão toca a corda (para baixo).

Especificação em [specs/](specs/): [requisitos](specs/requirements.md), [design](specs/design.md),
[tarefas](specs/tasks.md), [avaliação](specs/evaluation.md).

## Instalar (Windows, GPU NVIDIA, [uv](https://docs.astral.sh/uv/))

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt --override overrides.txt
curl -L -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

O `overrides.txt` impede o `opencv-python` (conflita com o `opencv-contrib-python` do MediaPipe).
O YOLOE é baixado sozinho na primeira execução.

## Usar

Dois cliques em **`GuitarHeroIRL.pyw`** (ou `.venv\Scripts\python -m ghirl`).

1. Braço inteiro do violão (com a cabeça) na imagem; espere o contorno verde.
2. Pestana na casa 1 → **feche e abra a outra mão**. Pestana na casa 12 → feche e abra.
3. Toque a nota que aparece (dedo na nota + a outra mão toca a corda para baixo). Ao acertar, vem a próxima.

Mão fechada por 3 s recomeça a calibração. No painel: botões, exercício, "trastes e notas" (T) e depuração (D).

Outros: `python -m ghirl --replay data/rec/violao_01` (sessão gravada), `--record data/rec/nova` (grava),
`--set secao.chave=valor` (muda a [config](config.yaml)). Testes: `.venv\Scripts\python -m pytest`.
