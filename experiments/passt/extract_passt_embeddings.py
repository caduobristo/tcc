from __future__ import annotations

import argparse
import csv
import json
import os
import re
import warnings
from pathlib import Path
from types import MethodType
from hear21passt.base import get_basic_model
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.manifold import TSNE


SAMPLE_RATE = 32000
DURATION_SECONDS = 10
TARGET_SAMPLES = SAMPLE_RATE * DURATION_SECONDS
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aac"}


def _load_audio_file(path: Path) -> tuple[torch.Tensor, int]:
    try:
        return torchaudio.load(str(path))
    except RuntimeError as exc:
        if "Numpy is not available" not in str(exc):
            raise

        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
        waveform = torch.tensor(data.T.tolist(), dtype=torch.float32)
        return waveform, sample_rate


def load_audio(path: Path) -> torch.Tensor:
    waveform, sample_rate = _load_audio_file(path)

    if waveform.numel() == 0:
        raise ValueError("audio vazio")

    waveform = waveform.float()

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sample_rate, SAMPLE_RATE)

    num_samples = waveform.shape[1]
    if num_samples > TARGET_SAMPLES:
        waveform = waveform[:, :TARGET_SAMPLES]
    elif num_samples < TARGET_SAMPLES:
        waveform = F.pad(waveform, (0, TARGET_SAMPLES - num_samples))

    return waveform.contiguous()


def _ensure_forward_features(model: torch.nn.Module) -> None:
    if hasattr(model, "forward_features"):
        return

    if not hasattr(model, "mel") or not hasattr(model, "net"):
        raise AttributeError("modelo nao expoe forward_features nem mel/net para compatibilidade")

    def forward_features(self: torch.nn.Module, audio: torch.Tensor) -> torch.Tensor:
        specs = self.mel(audio)
        specs = specs.unsqueeze(1)
        features = self.net.forward_features(specs)
        if isinstance(features, tuple):
            features = (features[0] + features[1]) / 2
        return features

    model.forward_features = MethodType(forward_features, model)


def extract_embedding(model: torch.nn.Module, audio: torch.Tensor) -> np.ndarray:
    audio = audio.to("cpu")
    with torch.no_grad():
        embedding = model.forward_features(audio)

    embedding = embedding.detach().cpu().reshape(-1)
    if embedding.numel() != 768:
        raise ValueError(f"embedding com tamanho inesperado: {embedding.numel()} (esperado: 768)")

    return np.asarray(embedding.tolist(), dtype=np.float32)


def infer_label(path: Path, label_token: int = 2, strip_label_digits: bool = True) -> str:
    parts = path.stem.split("_")
    label = parts[label_token] if 0 <= label_token < len(parts) else parts[0]
    if strip_label_digits:
        label = re.sub(r"\d+$", "", label)
    return label


def process_dataset(
    dataset_path: Path,
  label_token: int = 2,
  strip_label_digits: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset_path = Path(dataset_path)
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise FileNotFoundError(f"diretorio nao encontrado: {dataset_path}")

    audio_files = sorted(
        path for path in dataset_path.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        raise RuntimeError(f"nenhum arquivo de audio encontrado em {dataset_path}")

    print("Carregando modelo PaSST hear21passt em CPU...")
    model = get_basic_model(mode="logits")
    model.eval()
    model.to("cpu")
    _ensure_forward_features(model)

    embeddings: list[np.ndarray] = []
    labels: list[str] = []
    filenames: list[str] = []

    total = len(audio_files)
    for index, audio_path in enumerate(audio_files, start=1):
        label = infer_label(audio_path, label_token, strip_label_digits)
        print(f"[{index}/{total}] {audio_path.name} -> label={label}")
        try:
            audio = load_audio(audio_path)
            embedding = extract_embedding(model, audio)
        except Exception as exc:
            print(f"  ignorado: {exc}")
            continue

        embeddings.append(embedding)
        labels.append(label)
        filenames.append(audio_path.name)

    if not embeddings:
        raise RuntimeError("nenhum embedding foi extraido; verifique os arquivos de audio")

    return (
        np.stack(embeddings, axis=0),
        np.asarray(labels, dtype=str),
        np.asarray(filenames, dtype=str),
    )


def compute_tsne(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.shape[0] < 2:
        raise RuntimeError("t-SNE precisa de pelo menos 2 embeddings para visualizar")

    perplexity = min(30, max(1, embeddings.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Could not find the number of physical cores.*")
        return tsne.fit_transform(embeddings)


def visualize_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    filenames: np.ndarray | None = None,
    output_path: Path = Path("embeddings_tsne.png"),
    points: np.ndarray | None = None,
    annotate_ids: bool = True,
) -> np.ndarray:

    matplotlib.use("Agg")

    if points is None:
        points = compute_tsne(embeddings)

    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab10", len(unique_labels))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    for color_index, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            s=55,
            alpha=0.85,
            color=cmap(color_index),
            label=label,
            edgecolors="k",
            linewidths=0.3,
        )

    if annotate_ids:
        for point_id, (x_coord, y_coord) in enumerate(points):
            plt.annotate(
                str(point_id),
                (x_coord, y_coord),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=7,
                alpha=0.8,
            )

    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.title("Distribuicao dos embeddings PaSST")
    plt.legend(title="Label", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    return points


def _nearest_by_distance(distance_matrix: np.ndarray, item_index: int, neighbors_k: int) -> list[int]:
    distances = distance_matrix[item_index].copy()
    distances[item_index] = np.inf
    return np.argsort(distances)[:neighbors_k].tolist()


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-12)
    distances = 1.0 - np.matmul(normalized, normalized.T)
    np.fill_diagonal(distances, np.inf)
    return distances


def _euclidean_distance_matrix(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(distances, np.inf)
    return distances


def save_neighbors_csv(
    embeddings: np.ndarray,
    labels: np.ndarray,
    filenames: np.ndarray,
    points: np.ndarray,
    output_path: Path,
    neighbors_k: int = 5,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cosine_distances = _cosine_distance_matrix(embeddings)
    tsne_distances = _euclidean_distance_matrix(points)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "metric",
                "source_id",
                "source_file",
                "source_label",
                "neighbor_rank",
                "neighbor_id",
                "neighbor_file",
                "neighbor_label",
                "same_label",
                "distance",
            ]
        )
        for metric_name, distance_matrix in [
            ("embedding_cosine", cosine_distances),
            ("tsne_2d", tsne_distances),
        ]:
            for source_index, source_file in enumerate(filenames):
                neighbor_indices = _nearest_by_distance(distance_matrix, source_index, neighbors_k)
                for rank, neighbor_index in enumerate(neighbor_indices, start=1):
                    writer.writerow(
                        [
                            metric_name,
                            source_index,
                            source_file,
                            labels[source_index],
                            rank,
                            neighbor_index,
                            filenames[neighbor_index],
                            labels[neighbor_index],
                            labels[source_index] == labels[neighbor_index],
                            f"{distance_matrix[source_index, neighbor_index]:.8f}",
                        ]
                    )


def save_interactive_html(
    embeddings: np.ndarray,
    labels: np.ndarray,
    filenames: np.ndarray,
    points: np.ndarray,
    dataset_path: Path,
    output_path: Path,
    neighbors_k: int = 5,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cosine_distances = _cosine_distance_matrix(embeddings)
    tsne_distances = _euclidean_distance_matrix(points)

    records = []
    for index, filename in enumerate(filenames.tolist()):
        audio_path = Path(dataset_path) / filename
        audio_src = os.path.relpath(audio_path, start=output_path.parent).replace(os.sep, "/")
        records.append(
            {
                "id": index,
                "x": float(points[index, 0]),
                "y": float(points[index, 1]),
                "label": str(labels[index]),
                "filename": str(filename),
                "audio": audio_src,
                "embeddingNeighbors": [
                    {
                        "id": neighbor,
                        "distance": float(cosine_distances[index, neighbor]),
                    }
                    for neighbor in _nearest_by_distance(cosine_distances, index, neighbors_k)
                ],
                "tsneNeighbors": [
                    {
                        "id": neighbor,
                        "distance": float(tsne_distances[index, neighbor]),
                    }
                    for neighbor in _nearest_by_distance(tsne_distances, index, neighbors_k)
                ],
            }
        )

    payload = json.dumps(records, ensure_ascii=False)
    html = """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PaSST t-SNE interativo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #1d232f;
      --muted: #647084;
      --line: #d9dee8;
      --accent: #2364aa;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--ink);
    }
    main {
      display: grid;
      grid-template-columns: minmax(480px, 1fr) 390px;
      gap: 16px;
      min-height: 100vh;
      padding: 16px;
    }
    .plot-wrap, aside {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
    }
    .plot-wrap {
      position: relative;
      min-height: 680px;
      padding: 12px;
    }
    svg {
      width: 100%;
      height: calc(100vh - 56px);
      min-height: 640px;
      display: block;
    }
    aside {
      padding: 16px;
      overflow: auto;
      max-height: calc(100vh - 32px);
    }
    h1 {
      margin: 0 0 8px;
      font-size: 20px;
      line-height: 1.25;
    }
    h2 {
      margin: 18px 0 8px;
      font-size: 15px;
    }
    .hint, .meta {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .point {
      stroke: #172033;
      stroke-width: 1;
      cursor: pointer;
      opacity: 0.88;
    }
    .point:hover, .point.selected {
      stroke-width: 3;
      opacity: 1;
    }
    .axis {
      stroke: var(--line);
      stroke-width: 1;
    }
    .label-text {
      font-size: 11px;
      fill: #394150;
      pointer-events: none;
    }
    .tooltip {
      position: absolute;
      display: none;
      max-width: 320px;
      padding: 8px 10px;
      background: #172033;
      color: white;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.35;
      pointer-events: none;
      z-index: 5;
    }
    .selected-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fbfcfe;
    }
    .filename {
      overflow-wrap: anywhere;
      font-weight: 700;
      margin: 6px 0;
    }
    audio {
      width: 100%;
      height: 34px;
      margin-top: 8px;
    }
    .neighbor {
      border-top: 1px solid var(--line);
      padding: 10px 0;
    }
    .neighbor:first-child {
      border-top: 0;
    }
    .neighbor button {
      border: 1px solid var(--line);
      background: white;
      color: var(--accent);
      border-radius: 6px;
      padding: 4px 8px;
      cursor: pointer;
      font-size: 12px;
      margin-top: 6px;
    }
    .pill {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: #eef3fa;
      color: #223047;
      font-size: 12px;
      margin-right: 6px;
    }
    @media (max-width: 900px) {
      main {
        grid-template-columns: 1fr;
      }
      aside {
        max-height: none;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="plot-wrap">
      <svg id="plot" viewBox="0 0 1000 700" aria-label="t-SNE dos embeddings PaSST"></svg>
      <div id="tooltip" class="tooltip"></div>
    </section>
    <aside>
      <h1>Embeddings PaSST</h1>
      <p class="hint">Passe o mouse para ver arquivo e label. Clique em um ponto para ouvir o audio e comparar seus vizinhos.</p>
      <div id="details"></div>
    </aside>
  </main>
  <script>
    const data = __PAYLOAD__;
    const colors = ["#2364aa", "#d95f02", "#2a9d8f", "#b23a48", "#6a4c93", "#f4a261", "#4d908e", "#bc6c25", "#577590", "#8ab17d"];
    const labels = [...new Set(data.map(d => d.label))].sort();
    const colorByLabel = Object.fromEntries(labels.map((label, index) => [label, colors[index % colors.length]]));
    const svg = document.getElementById("plot");
    const tooltip = document.getElementById("tooltip");
    const details = document.getElementById("details");
    const margin = { left: 62, right: 24, top: 24, bottom: 54 };
    const width = 1000;
    const height = 700;
    const xs = data.map(d => d.x);
    const ys = data.map(d => d.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const padX = (maxX - minX || 1) * 0.08;
    const padY = (maxY - minY || 1) * 0.08;

    function sx(x) {
      return margin.left + ((x - minX + padX) / (maxX - minX + padX * 2 || 1)) * (width - margin.left - margin.right);
    }
    function sy(y) {
      return height - margin.bottom - ((y - minY + padY) / (maxY - minY + padY * 2 || 1)) * (height - margin.top - margin.bottom);
    }
    function makeSvg(tag, attrs = {}) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([key, value]) => el.setAttribute(key, value));
      return el;
    }
    function makeEl(tag, className, text) {
      const el = document.createElement(tag);
      if (className) el.className = className;
      if (text !== undefined) el.textContent = text;
      return el;
    }
    function formatDistance(value) {
      return Number(value).toFixed(4);
    }
    function renderPlot() {
      svg.replaceChildren();
      svg.appendChild(makeSvg("line", { x1: margin.left, y1: height - margin.bottom, x2: width - margin.right, y2: height - margin.bottom, class: "axis" }));
      svg.appendChild(makeSvg("line", { x1: margin.left, y1: margin.top, x2: margin.left, y2: height - margin.bottom, class: "axis" }));

      data.forEach(d => {
        const circle = makeSvg("circle", {
          cx: sx(d.x),
          cy: sy(d.y),
          r: 7,
          fill: colorByLabel[d.label],
          class: "point",
          "data-id": d.id,
        });
        circle.addEventListener("mousemove", event => {
          tooltip.style.display = "block";
          tooltip.style.left = `${event.offsetX + 14}px`;
          tooltip.style.top = `${event.offsetY + 14}px`;
          tooltip.textContent = `#${d.id} | ${d.label} | ${d.filename}`;
        });
        circle.addEventListener("mouseleave", () => {
          tooltip.style.display = "none";
        });
        circle.addEventListener("click", () => selectPoint(d.id));
        svg.appendChild(circle);

        const idText = makeSvg("text", { x: sx(d.x) + 9, y: sy(d.y) + 4, class: "label-text" });
        idText.textContent = d.id;
        svg.appendChild(idText);
      });
    }
    function renderNeighborList(title, neighbors, distanceLabel) {
      const section = makeEl("section");
      section.appendChild(makeEl("h2", null, title));
      neighbors.forEach((neighborInfo, index) => {
        const item = data[neighborInfo.id];
        const row = makeEl("div", "neighbor");
        const line = makeEl("div");
        line.appendChild(makeEl("span", "pill", `#${item.id}`));
        line.appendChild(makeEl("span", "pill", item.label));
        line.appendChild(document.createTextNode(` ${index + 1}. ${item.filename}`));
        row.appendChild(line);
        row.appendChild(makeEl("div", "meta", `${distanceLabel}: ${formatDistance(neighborInfo.distance)}`));
        const button = makeEl("button", null, "selecionar no grafico");
        button.addEventListener("click", () => selectPoint(item.id));
        row.appendChild(button);
        const audio = makeEl("audio");
        audio.controls = true;
        audio.src = item.audio;
        row.appendChild(audio);
        section.appendChild(row);
      });
      return section;
    }
    function selectPoint(id) {
      const selected = data[id];
      document.querySelectorAll(".point").forEach(point => {
        point.classList.toggle("selected", Number(point.dataset.id) === id);
      });
      details.replaceChildren();
      const card = makeEl("div", "selected-card");
      card.appendChild(makeEl("div", "meta", `Ponto #${selected.id}`));
      card.appendChild(makeEl("div", "filename", selected.filename));
      const label = makeEl("div");
      label.appendChild(makeEl("span", "pill", selected.label));
      label.appendChild(makeEl("span", "meta", `x=${formatDistance(selected.x)} | y=${formatDistance(selected.y)}`));
      card.appendChild(label);
      const audio = makeEl("audio");
      audio.controls = true;
      audio.src = selected.audio;
      card.appendChild(audio);
      details.appendChild(card);
      details.appendChild(renderNeighborList("Vizinhos no embedding original", selected.embeddingNeighbors, "distancia cosseno"));
      details.appendChild(renderNeighborList("Vizinhos proximos no t-SNE", selected.tsneNeighbors, "distancia 2D"));
    }
    renderPlot();
    selectPoint(0);
  </script>
</body>
</html>
"""
    output_path.write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai embeddings PaSST hear21passt de audios em dataset/ e gera t-SNE."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--input-npz", type=Path, default=None)
    parser.add_argument("--output-npz", type=Path, default=Path("embeddings.npz"))
    parser.add_argument("--output-plot", type=Path, default=Path("embeddings_tsne.png"))
    parser.add_argument("--output-html", type=Path, default=Path("embeddings_tsne_interactive.html"))
    parser.add_argument("--output-neighbors", type=Path, default=Path("embeddings_neighbors.csv"))
    parser.add_argument("--neighbors-k", type=int, default=5)
    parser.add_argument("--label-token", type=int, default=2)
    parser.add_argument("--strip-label-digits", action="store_true", default=True)
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--skip-html", action="store_true")
    parser.add_argument("--skip-neighbors", action="store_true")
    parser.add_argument("--no-annotate-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_npz:
        data = np.load(args.input_npz)
        embeddings = data["embeddings"]
        labels = data["labels"]
        filenames = data["filenames"]
        if args.label_token != 0 or args.strip_label_digits:
            labels = np.asarray(
                [
                    infer_label(Path(filename), args.label_token, args.strip_label_digits)
                    for filename in filenames
                ],
                dtype=str,
            )
        print(f"Embeddings carregados de: {args.input_npz}")
    else:
        embeddings, labels, filenames = process_dataset(
            args.dataset,
            label_token=args.label_token,
            strip_label_digits=args.strip_label_digits,
        )

        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.output_npz,
            embeddings=embeddings,
            labels=labels,
            filenames=filenames,
        )
        print(f"Embeddings salvos em: {args.output_npz}")

    print(f"Shape embeddings: {embeddings.shape}")

    points = compute_tsne(embeddings)

    if not args.skip_plot:
        try:
            visualize_embeddings(
                embeddings,
                labels,
                filenames=filenames,
                output_path=args.output_plot,
                points=points,
                annotate_ids=not args.no_annotate_plot,
            )
            print(f"Visualizacao salva em: {args.output_plot}")
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib":
                raise
            print("Visualizacao PNG ignorada: matplotlib nao esta instalado.")

    if not args.skip_html:
        save_interactive_html(
            embeddings,
            labels,
            filenames,
            points,
            args.dataset,
            args.output_html,
            neighbors_k=args.neighbors_k,
        )
        print(f"Visualizacao interativa salva em: {args.output_html}")

    if not args.skip_neighbors:
        save_neighbors_csv(
            embeddings,
            labels,
            filenames,
            points,
            args.output_neighbors,
            neighbors_k=args.neighbors_k,
        )
        print(f"Vizinhos salvos em: {args.output_neighbors}")


if __name__ == "__main__":
    main()
