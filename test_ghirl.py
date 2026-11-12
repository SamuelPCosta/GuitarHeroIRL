"""Testes do Guitar Hero IRL: um por parte do sistema, com dados sintéticos (sem câmera, sem GPU)."""

import json
import time

import cv2
import numpy as np
import pytest

from ghirl.app import Pipeline, main
from ghirl.calib import (
    AxisCalibration,
    Calibrator,
    NeckGeometry,
    finger_g,
    fret_g,
    solve_affine,
    transfer_calibration,
)
from ghirl.capture import Frame, Recorder
from ghirl.config import DEFAULT_CONFIG_PATH, CalibConfig, Config, ConfigError, GameConfig, NeckConfig, load_config
from ghirl.game import Exercise, ExerciseError, Game, Note, PluckDetector, list_exercises
from ghirl.geometry import AxisFrame, Keyframe, Line, NeckDetection, NeckPose, apply_homography, undirected_angle_deg
from ghirl.hands import GestureDetector, Hand
from ghirl.neck import SegWorker, analyze_mask
from ghirl.tracking import FeatureTracker, NeckEstimator

W, H = 1280, 720

# ---------------------------------------------------------------------------------------- cenas sintéticas

# Braço horizontal: eixo y = 300, bordas em 276 (bordões) e 324, cabeça à direita.
# Calibração verdadeira s = -650 g + 500 (nut em x = 1000, traste 12 em x = 675).
TRUE = AxisCalibration(a=-650.0, b=500.0)


def make_geom():
    det = NeckDetection(0, None, AxisFrame.make([500, 300], [1, 0]), Line.make([0, 276], [1, 0]),
                        Line.make([0, 324], [1, 0]), extent=(-200, 520), nut_side_hint=+1)
    return NeckGeometry.from_detection(det, "top")


def barre_hand(fret):
    """Mão com o indicador esticado atravessando o braço na casa ``fret``."""
    x = 500 + float(TRUE.s_of_g(finger_g(fret, 0.75)))
    pts = np.tile([x + 40, 380.0], (21, 1))
    for i, dy in zip((5, 6, 7, 8), (0, -25, -45, -62), strict=True):
        pts[i] = [x, 340 + dy]
    return Hand(pts.astype(np.float32))


def tip_hand(point):
    """Mão com a ponta do indicador em ``point`` e o resto longe do braço."""
    pts = np.tile([900.0, 650.0], (21, 1))  # palma perto da cabeça, fora do braço
    pts[8] = point
    return Hand(pts.astype(np.float32))


def string_y(k, margin=0.09):
    """Altura (y) da corda k no braço sintético (bordas em y = 276 e 324, corda 6 em cima)."""
    return 276 + 48 * (margin + (6 - k) / 5 * (1 - 2 * margin))


def pluck_hand(y):
    """Mão sobre o corpo do violão (g ≈ 0.75, além da casa 12) com a ponta do indicador na altura y."""
    x = 500 + float(TRUE.s_of_g(0.75))
    pts = np.tile([x, 380.0], (21, 1))
    pts[8] = [x, y]
    return Hand(pts.astype(np.float32))


def synth_guitar(angle_deg):
    """Máscara de violão (corpo + braço + cabeça em +x), girada. Devolve (máscara, direção do braço)."""
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])

    def tf(p):
        return (np.asarray(p, float) @ R.T + [640, 360]).astype(np.int32)

    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [tf([[-150, -27], [230, -21.5], [230, 21.5], [-150, 27]])], 255)  # braço
    cv2.fillPoly(m, [tf([[230, -40], [340, -40], [340, 40], [230, 40]])], 255)  # cabeça
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    for cx, r in ((-260, 120), (-400, 130)):  # corpo
        cv2.fillPoly(m, [tf(np.stack([cx + r * np.cos(t), r * np.sin(t)], 1))], 255)
    return m, R @ [1.0, 0.0]


REGION = np.array([[250, 260], [1050, 230], [1060, 450], [240, 470]], np.int32)


def texture():
    rng = np.random.default_rng(0)
    img = np.full((H, W, 3), 90, np.uint8)
    for _ in range(900):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        cv2.circle(img, (x, y), int(rng.integers(2, 9)), c, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def textured_keyframe(img):
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [REGION], 255)
    det = NeckDetection(0, mask, AxisFrame.make([650, 350], [1, 0]), Line.make([0, 260], [1, 0]),
                        Line.make([0, 450], [1, 0]), (-400, 400), confidence=0.9, full_mask=mask)
    return Keyframe(0, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), det, mask)


def motion(k, n=40):
    """Homografia verdadeira: rotação, zoom, translação e um pouco de perspectiva, suaves."""
    f = k / n
    a = np.radians(12) * f
    T1 = np.array([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1]])
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    S = np.diag([1 + 0.1 * f, 1 + 0.1 * f, 1])
    P = np.array([[1, 0, 0], [0, 1, 0], [2e-5 * f, 0, 1]])
    T2 = np.array([[1, 0, W / 2 + 70 * f], [0, 1, H / 2 - 40 * f], [0, 0, 1]])
    return T2 @ P @ S @ R @ T1


def corner_error(Ha, Hb):
    return float(np.mean(np.linalg.norm(apply_homography(Ha, REGION) - apply_homography(Hb, REGION), axis=1)))


# ---------------------------------------------------------------------------------------- testes


def test_config_yaml_matches_defaults_and_rejects_bad_keys():
    assert load_config(DEFAULT_CONFIG_PATH).to_dict() == Config().to_dict()
    assert load_config(None, ["camera.fps=15"]).camera.fps == 15
    for bad in ("camera.nada=1", "display.mirror=1", "calib.mode=9p"):
        with pytest.raises(ConfigError):
            load_config(None, [bad])


def test_mask_gives_neck_axis_edges_and_head_side():
    mask, u = synth_guitar(40.0)
    det = analyze_mask(mask)
    assert det is not None and undirected_angle_deg(det.axis.u, u) < 2.0
    assert det.nut_side_hint == (1 if np.dot(det.axis.u, u) > 0 else -1)  # cabeça em +x
    assert det.width_at(0.0) == pytest.approx(48.5, abs=4)


def test_tracker_follows_motion_and_ignores_hand():
    base = texture()
    trk = FeatureTracker()
    trk.init(Frame(base, 0.0, 0), textured_keyframe(base))
    for k in range(1, 41):
        img = cv2.warpPerspective(base, motion(k), (W, H), borderValue=(90, 90, 90))
        x = 300 + 15 * k  # "mão" passando por cima
        cv2.rectangle(img, (x, 220), (x + 160, 480), (40, 120, 200), -1)
        excl = np.zeros((H, W), np.uint8)
        cv2.rectangle(excl, (x - 15, 205), (x + 175, 495), 255, -1)
        pose = trk.update(Frame(img, k / 30, k), excl)
        assert pose.status != "perdido" and corner_error(pose.H, motion(k)) < 2.5


def test_estimator_finds_loses_and_relocalizes_same_keyframe():
    base, blank = texture(), np.full((H, W, 3), 90, np.uint8)

    class Locator:
        def detect(self, image, frame_index=0):
            return textured_keyframe(base).detection if image.std() > 3 else None

    worker = SegWorker(Locator)
    worker.start()
    est = NeckEstimator(NeckConfig(lost_frames=3), worker)
    i = 0

    def feed(img, until):
        nonlocal i
        t0 = time.perf_counter()
        while not until() and time.perf_counter() - t0 < 5:
            i += 1
            est.update(Frame(img, i / 30, i))
            time.sleep(0.005)
        return until()

    try:
        assert feed(base, lambda: est.status == "rastreando")
        kid = est.keyframe_id
        assert feed(blank, lambda: est.status == "perdido")
        for _ in range(240):  # 8 s fora do quadro: continua esperando o MESMO quadro-chave
            i += 1
            est.update(Frame(blank, i / 30, i))
        assert est.status == "perdido" and est.keyframe is not None
        moved = cv2.warpPerspective(base, motion(10), (W, H), borderValue=(90, 90, 90))
        assert feed(moved, lambda: est.status == "rastreando") and est.keyframe_id == kid
        assert corner_error(est.pose.H, motion(10)) < 3.0
    finally:
        worker.stop()


def test_calibration_by_click():
    cal = Calibrator(CalibConfig(), make_geom(), keyframe_id=1)
    assert cal.capture() is not None  # sem pestana, o clique não captura
    t = 0.0
    for fret in (1, 1, 12):
        for _ in range(30):  # segurar por 1 s NÃO captura sozinho
            cal.observe(t, [barre_hand(fret)], np.eye(3))
            t += 1 / 30
        err = cal.capture()
        if len(cal.measures) == 1 and fret == 1 and err:
            assert "mesma posição" in err  # 2º clique ainda na casa 1 é recusado
        t += 2.0  # o usuário troca de casa no tempo dele
    assert cal.done
    frets = fret_g(np.arange(13))
    assert np.abs(cal.model.calibration.s_of_g(frets) - TRUE.s_of_g(frets)).max() < 2.0


def test_frets_in_image_refine_the_scale():
    """Com perspectiva (c != 0) as pestanas sozinhas erram; os trastes visíveis corrigem."""
    true = AxisCalibration(a=-700.0, b=500.0, c=0.25)
    gray = np.full((H, W), 40, np.uint8)
    for s in true.fret_s(12):
        cv2.line(gray, (int(round(500 + s)), 278), (int(round(500 + s)), 322), 200, 2)
    barres = [(n, float(true.s_of_g(finger_g(n, 0.75)))) for n in (1, 12)]
    cal = Calibrator(CalibConfig(), make_geom(), 1, gray_K=gray)
    cal.measures = barres[:1]
    cal._recent.extend([(0, barres[1][1])] * 3)
    assert cal.capture() is None and len(cal.frets_found) >= 10
    frets = fret_g(np.arange(13))
    before = np.abs(solve_affine(barres, 0.75).s_of_g(frets) - true.s_of_g(frets)).max()
    after = np.abs(cal.model.calibration.s_of_g(frets) - true.s_of_g(frets)).max()
    assert before > 4 and after < 1.5

    # quadro-chave novo (violão girado, mais perto e deslocado): a calibração vai junto e é
    # reajustada pelos trastes da imagem nova, sem pestanas
    a = np.radians(10)
    Hk = np.array([[1.1 * np.cos(a), -1.1 * np.sin(a), 60], [1.1 * np.sin(a), 1.1 * np.cos(a), -40], [0, 0, 1]])
    gray2 = cv2.warpPerspective(gray, Hk, (W, H), borderValue=40)
    d1 = make_geom().detection
    o2 = apply_homography(Hk, [d1.axis.origin])[0]
    u2 = apply_homography(Hk, [d1.axis.origin + d1.axis.u])[0] - o2
    d2 = NeckDetection(0, None, AxisFrame.make(o2, u2), d1.edge_a.transformed(Hk), d1.edge_b.transformed(Hk),
                       extent=(-220, 572), nut_side_hint=+1)
    geom2 = NeckGeometry.from_detection(d2, "top")
    shift = np.array([[1.0, 0, 5], [0, 1, 0], [0, 0, 1]])  # registro entre os quadros-chave com 5 px de erro
    moved, found = transfer_calibration(cal.model, Hk @ shift, geom2, gray2, 2, 0.09)
    want = apply_homography(Hk, d1.axis.to_image(np.stack([true.fret_s(12), np.zeros(13)], 1)))
    got = moved.geometry.detection.axis.to_image(np.stack([moved.calibration.fret_s(12), np.zeros(13)], 1))
    assert len(found) >= 10 and np.linalg.norm(got - want, axis=1).max() < 2.0
    shift[0, 2] = 25  # meia casa de erro: ambíguo, recusa em vez de errar a casa
    assert transfer_calibration(cal.model, Hk @ shift, geom2, gray2, 2, 0.09) is None


def test_gestures_close_open_and_hold():
    def hand(tip_ratio):
        pts = np.zeros((21, 2), np.float32) + [400, 300]
        for mcp, tip, dx in ((5, 8, -30), (9, 12, -10), (13, 16, 10), (17, 20, 30)):
            pts[mcp] = [400 + dx, 250]
            pts[tip] = [400 + dx * tip_ratio, 300 - 50 * tip_ratio]
        return Hand(pts)

    opened, fist = hand(2.0), hand(1.0)
    g = GestureDetector(advance_max_s=1.5, reset_hold_s=3.0)
    seq = [opened] * 5 + [fist] * 10 + [opened] * 5
    assert [e for i, h in enumerate(seq) if (e := g.update(i / 30, [h]))] == ["avancar"]
    events = [g.update(1 + i / 30, [fist]) for i in range(100)] + [g.update(5 + i / 30, [opened]) for i in range(5)]
    assert [e for e in events if e] == ["reiniciar"]  # segurar não vira "avançar" ao abrir


def test_exercises_json_and_sequential_game(tmp_path):
    found = list_exercises(load_config().game.exercises_dir)
    assert Exercise.load(found["pentatonica_am_casa5"]).notes[:2] == [Note(6, 5), Note(6, 8)]
    (tmp_path / "ruim.json").write_text(json.dumps({"notas": [{"corda": 7, "casa": 5}]}), encoding="utf-8")
    with pytest.raises(ExerciseError):
        Exercise.load(tmp_path / "ruim.json")

    g = Game(Exercise("t", [Note(6, 5), Note(5, 7)]), GameConfig(hold_frames=2))
    on_first = np.array([[4.6, 6.1]])  # casa 5 = (4, 5], corda ~6
    assert not any(g.update(i / 30, on_first) for i in range(10))  # só o dedo, sem tocar a corda: não vale
    assert g.update(0.4, on_first, pluck=True) and g.current == Note(5, 7)
    assert not g.update(0.5, on_first, pluck=True)  # a nota antiga não vale mais

    def run(det, ks, target, t0):
        return any(det.update(t0 + i / 30, np.array([k, 9.0]), target) for i, k in enumerate(ks))

    pl = PluckDetector("down")
    assert not run(pl, [3.1, 3.0, 2.95, 3.05, 3.0], 3, 0)  # dedo parado na corda (tremidinha)
    assert run(pl, [3.2, 3.0, 2.7, 2.4], 3, 1)  # desceu pela corda 3
    assert not run(pl, [2.4, 2.7, 3.0, 3.3], 3, 2)  # subindo não vale
    assert not run(pl, [5.2, 5.0, 4.6, 4.3], 3, 3)  # desceu por outra corda
    assert run(PluckDetector("any"), [2.4, 2.7, 3.0, 3.6], 3, 4)


class FakeNeck:
    """Braço sempre rastreado com H = identidade."""

    status, keyframe_id, last_detection_ms = "rastreando", 1, 0.0

    def __init__(self):
        self.keyframe = Keyframe(0, np.zeros((H, W), np.uint8), make_geom().detection, np.zeros((H, W), np.uint8))

    def update(self, frame, mask):
        return NeckPose(frame.index, np.eye(3), status="ok")


def test_pipeline_calibrate_then_one_note_at_a_time():
    class Hands:
        next: list = []

        def detect(self, frame):
            return self.next

    hands = Hands()
    pipe = Pipeline(load_config(None, ["game.hold_frames=2"]), hands, FakeNeck())
    img = np.zeros((H, W, 3), np.uint8)
    t = [0]

    def step(n=1):
        for _ in range(n):
            t[0] += 1
            out = pipe.process(Frame(img, t[0] / 30, t[0]))
        return out

    for fret in (1, 12):
        hands.next = [barre_hand(fret)]
        step(5)
        assert pipe.state == "calibrando"
        pipe.capture_barre()
    assert pipe.state == "jogando"

    hands.next = []
    out = step()
    note = pipe.game.current
    c = pipe.model.cell_center_K(note.string, note.fret)
    ys, xs = np.nonzero(out.any(axis=2))
    assert np.abs(xs - c[0]).max() < 60 and np.abs(ys - c[1]).max() < 40  # só a nota atual, sem grade

    n = pipe.game.current
    hands.next = [tip_hand(pipe.model.cell_center_K(n.string, n.fret))]
    step(10)
    assert pipe.game.index == 0  # dedo na nota, mas a outra mão não tocou a corda
    for _ in range(3):
        n = pipe.game.current
        for dy in (-3, 0, 4, 8):  # um dedo da outra mão desce pela corda certa, sobre o corpo
            hands.next = [tip_hand(pipe.model.cell_center_K(n.string, n.fret)), pluck_hand(string_y(n.string) + dy)]
            step()
        hands.next = []
        step()
    assert pipe.game.index == 3


def test_cli_headless_replay(tmp_path, capsys):
    with Recorder(tmp_path / "rec", fps=30, size=(320, 180)) as rec:
        for i in range(15):
            rec.write(Frame(np.zeros((180, 320, 3), np.uint8), i / 30, i))
    assert main(["--replay", str(tmp_path / "rec"), "--replay-mode", "sequential", "--headless", "--no-hands",
                 "--no-neck", "--set", "log.jsonl=true", "--set", f"log.dir={(tmp_path / 'logs').as_posix()}"]) == 0
    assert "15 quadros" in capsys.readouterr().out
    (log,) = (tmp_path / "logs").glob("*.jsonl")
    assert len(log.read_text(encoding="utf-8").splitlines()) == 15
