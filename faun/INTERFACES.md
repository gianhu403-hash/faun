# Faun pipeline — immutable interface spec (v2)

This contract is FROZEN. All Phase-2 code waves write against it; do not change signatures.

```python
# faun/ingest: scan(path: Path) -> Manifest
#   AudioFileEntry(path, trap_id, start_dt, lat, lon, meta: dict, duration_s, sr)
#   info.txt (CSV: date,time,long,lat,battery,temp,humidity,filename,sample_rate,gain,channel)
#   + timestamp-from-filename parsed by ingest. One folder per trap (A1..A5), each with info.txt.
# faun/segmentation: SegmentExtractor.extract(waveform, sr) -> list[Segment]
#   Segment(start_s, duration_s); internally downmix mono + resample 48k->16k -> onset.py
# faun/classification:
#   class SpeciesClassifier(Protocol): def classify(self, segment, sr) -> list[Prediction]: ...
#   Prediction(species: str, probability: float)
#   StubAdapter (in skeleton), BirdNETAdapter, YAMNetAdapter (embeddings+probe, NOT anthropic head), PerchAdapter
# faun/jobs: Job(job_id: uuid, workdir=jobs_root/<job_id>/, status, manifest.json, results.csv)
#   batch isolation = namespace per job_id, no shared temp paths
# faun/output: CsvWriter -> columns: track, start_sec, duration_sec, species, probability (+ sidecar trap metadata)
# faun/storage: Storage(Protocol: put/get/url) -> only LocalFSStorage (S3 is a July task, NOT now)
# faun/api: POST /jobs {source_path|url, lat, lon}->{job_id}; GET /jobs/{id}->status; GET /jobs/{id}/results.csv
# faun/cli: faun process <dir> [--out results.csv]
```

UI (faun/static/index.html, vanilla JS, single file): form (folder/URL) -> POST /jobs -> poll status with progress -> table + download CSV.

## v2 run-ready additions (ADDITIVE — every frozen signature above is unchanged)

Added in the run-ready iteration. These extend the contract additively; they do not
alter any signature in the FROZEN block above.

```python
# faun/embeddings: the SINGLE owner of batch embedding-export (no dupes elsewhere)
class Embedder(Protocol): def embed(self, waveform: np.ndarray, sr: int) -> np.ndarray   # fixed-dim vector
class PerchEmbedder:  DIM = 1280   # wraps experiments.wrappers.perch; downmix->32k->pad/truncate 160000
class YamnetEmbedder: DIM = 2048   # wraps experiments.wrappers.yamnet_probe; downmix->16k->concat(mean,max)
def embed_batch(clips: Iterable[tuple[np.ndarray, int]], embedder: Embedder) -> np.ndarray   # [N, DIM]
class EmbeddingCache(embeddings, ids=None, labels=None): save(path)->Path; @classmethod load(path)->EmbeddingCache

# faun/datasets: iNatSounds — first dataset with TRUE species labels (root/<species>/<clip>)
class iNatRecord(path: str, species: str)
class iNatSoundsDataset(root): manifest()->list[iNatRecord]; vocab()->dict[str,int]; split(seed)->(train, val)

# faun/retraining: ADDED (existing train_probe_cv/save_probe/load_probe/retrain_from_labels unchanged)
def species_eval(clf, X, y, *, synthetic: bool = True) -> dict
#   keys: per_species_recall, macro_f1, confusion, labels, n, n_classes, provenance, metric, value, ci_low, ci_high, note
#   provenance == "SYNTHETIC — not a species metric" when synthetic=True (honesty gate)

# faun/labeling: multi-model pseudo-labeling
def batch_label(archive, models: Mapping[str, SpeciesClassifier], out_jsonl, emb_out=None, embedder=None) -> dict
def training_candidates(detections) -> list   # CC BY-NC-SA gate: drops model:birdnet labels (never in training)

# faun/health
def health() -> dict   # {status: ok|degraded, service: "faun-api", version, jobs_root_writable}

# faun/api: GET /healthz -> health()
# faun/cli (additive subcommands):
#   faun batch-label --archive <dir> --out <jsonl> [--emb-out <npz>] [--embedder perch|yamnet] [--models perch,birdnet]
#   faun fetch-dataset --root <iNatSounds dir>
#   faun eval-species --probe <pkl> --dataset <dir> [--embedder perch|yamnet] [--seed 42]
```

## v2.1 additions — URL/Я.Диск ingest, Perch 2, fine-tune, settings/obs (ADDITIVE)

Added by the perch2-finetune-ingest wave. Every FROZEN signature above is unchanged;
the pipeline is extended only with new modules, a new sidecar field, and new CLI args.

```python
# faun/sources: resolve a source (local path / http(s) zip / Yandex.Disk share) to a
#   LOCAL directory for faun.ingest.scan — the P0 fix (no more Path("https://...")).
class SourceError(RuntimeError)   # .kind in {bad-scheme,ssrf,not-found,network,too-large,zip-slip,not-an-archive,empty}
def resolve_source(src: str, workdir: Path, *, client=None) -> Path   # local -> Path(src); remote -> download+safe-extract under workdir/_source/
def source_provenance(src: str) -> dict   # {"source": src, "mode": "local"|"http"|"yadisk"}
#   Я.Диск: public_key = share root, subfolder via &path=/A1 (NOT folded into public_key).
#   SSRF: getaddrinfo IP-resolve, rejects private/loopback/link-local + CGNAT 100.64/10, re-checked after redirects.
#   Limits via env FAUN_SOURCE_{TIMEOUT_S,MAX_BYTES,MAX_UNCOMPRESSED_BYTES,MAX_ENTRIES,MAX_REDIRECTS}.

# faun/classification: Perch 2 (Apache 2.0) — 1536-dim embeddings + ~14.8k species logits.
class Perch2Adapter   # classify(segment, sr)->list[Prediction]; embed(waveform, sr)->np.ndarray[1536]; DIM=1536
#   PERCH_V2_DIM = 1536 (NOT Perch-1's 1280). 32 kHz mono, 5 s = 160000 samples.
#   kagglehub.model_download -> tf.saved_model serving_default (NOT Perch-1 infer_tf). Lazy TF (>=2.20) + kagglehub.
#   Source: model_path -> PERCH_V2_MODEL_PATH -> kagglehub. NO creds AND no path -> RuntimeError (never falls back to Perch 1).
#   detections source tag SOURCE_PERCH_V2 = "model:perch-v2". FAUN_CLASSIFIER=perch-v2.
# faun/embeddings: class Perch2Embedder: DIM = 1536   # downmix->32k->pad/truncate 160000

# faun/training: REAL PyTorch audio-transformer fine-tune on iNatSounds (distinct from the frozen probe).
class iNatTorchDataset(root, vocab=None, *, records=None, sr=32000, win_s=10.0)   # (waveform, label_idx)
def make_loaders(root, vocab, *, seed, sr=32000, win_s=10.0, batch_size=16, num_workers=0) -> (DataLoader, DataLoader)
class Backbone(Protocol): feature_dim: int; def forward(self, batch): ...
def build_backbone(name="passt", *, sr=32000, win_s=10.0, freeze=True) -> Backbone   # passt(768)/ast/beats/stub
class SpeciesHead(feature_dim, n_classes)
def finetune(dataset_root, *, vocab=None, model="passt", out, epochs=15, batch_size=16, lr=3e-4,
             device="auto", amp=True, grad_accum=2, freeze_epochs=3, patience=4, class_weight=True,
             seed=42, resume=None, _backbone=None, _loaders=None) -> dict   # NO module-level torch
def save_checkpoint(...) / load_checkpoint(...)   # {state_dict, vocab, model_name, feature_dim, sr, clip_sec, provenance, epoch}
#   HONESTY: real species metric only after scripts/finetune_inatsounds.sh on cluster.

# faun/settings: centralized typed config (single source for jobs_root, classifier, model paths, log_json).
@dataclass(frozen=True) class Settings; def get_settings() -> Settings   # lru_cache; cache_clear() in tests
# faun/obs: def setup_logging(json=True); with_job_context(job_id)   # structured JSON logs, stdlib only

# faun/api: FAUN_CLASSIFIER also accepts "perch-v2"; results_meta.json gains a "source_provenance" object;
#   a failed job carries job.params["error_kind"] (SourceError.kind) alongside "error".
# faun/cli (additive): faun finetune --dataset <root> --out <ckpt_dir> [--model passt|ast|beats] [...];
#   --embedder now also accepts perch-v2 (batch-label / eval-species).
```

## v3 additions — audio/pipeline owners, Basic-Auth, ops (ADDITIVE)

Added by the prod-ready wave. Every FROZEN signature above is unchanged: run_pipeline and
batch_label keep their exact signatures (they became thin wrappers over faun.pipeline.run_batch);
ingest.scan, CSV columns and detections.jsonl are untouched. These are new modules + additive knobs.

```python
# faun/audio: the SINGLE owner of audio preprocessing (ADR-0002). numpy + soxr only.
def downmix(waveform: np.ndarray) -> np.ndarray            # stereo/multi -> mono float32
def resample(mono: np.ndarray, sr: int, target_sr: int) -> np.ndarray   # soxr (linear fallback); sr<=0 -> ValueError
def fit_window(mono: np.ndarray, win_samples: int) -> np.ndarray        # pad-right or truncate to exactly win_samples
#   faun.embeddings re-exports these as _downmix/_resample/_fit_window — SAME objects
#   (frozen import in faun/training/dataset.py; invariant: faun.embeddings._downmix is faun.audio.downmix).

# faun/pipeline: reusable executor for the shared segment->classify->Detection core (ADR-0003).
CLASSIFY_SR = 16000   # classifier-input contract (mono float32 @ 16 kHz, NOT a Segment)
def slice_clip(waveform, sr, segment) -> np.ndarray         # clip on ORIGINAL sr/channels (bounds clamped)
def to_classifier_input(clip, sr) -> np.ndarray             # downmix + resample to 16 kHz mono (via faun.audio)
@dataclass class SegmentResult(detection: Detection, clip: np.ndarray, sr: int)
def run_batch(entries, *, read_waveform, build_labels, extractor=None) -> Iterator[SegmentResult]
#   GENERATOR; yields one SegmentResult per detected segment with clip ROW-ALIGNED to detection
#   (the contract batch_label's embeddings export relies on). run_pipeline + batch_label use it.

# faun/settings: + basic_user / basic_pass (None|str), read VERBATIM (no strip) via _env_secret_opt
#   from FAUN_BASIC_USER / FAUN_BASIC_PASS. get_settings() is now the actual reader across
#   api/sources/adapters/health.
# faun/health: FAUN_VERSION sourced from os.environ (default "2.0.0-rc") so /healthz reflects the build.

# faun/api: env-gated HTTP Basic Auth middleware. Both FAUN_BASIC_USER+PASS set -> every path
#   EXCEPT /healthz requires Basic creds (hmac.compare_digest; 401 + WWW-Authenticate); either unset
#   -> default-OPEN (no behaviour change). JobRequest.lat/lon get WGS84 Field bounds (reject NaN/Inf/
#   out-of-range -> 422); LabelRequest.species/source get length bounds. run_pipeline + batch_label
#   remove <workdir>/_source on exit (finally) — the remote-ingest disk-leak guard.
```

## v5 additions — real Perch 2 labels, Perch-probe serving, label export, TF image (ADDITIVE)

Added by the perch2-live wave (real Perch 2 on prod). Every FROZEN signature above is
unchanged; only new methods, a new adapter, a new CLI subcommand and a new deploy
artifact are introduced.

```python
# faun/classification/perch_v2: Perch2Adapter now names predictions with REAL species.
#   _load_labels() -> list[str] | None   # lazily reads <model_path>/assets/labels.csv,
#     DROPS the leading taxonomy header line (e.g. "inat2024_fsd50k"); the rest are
#     scientific names aligned 1:1 with the 'label' logits (verified real file = 14795).
#     Missing/garbage assets -> None -> classify() falls back to species_<i> (never crashes);
#     result cached; pure file I/O (TF-free). classify() guards out-of-range logit indices.
#     PERCH_V2_LABELS_FILE = "labels.csv".

# faun/classification/perch_probe: Perch 2 embeddings + a trained probe head (served).
class PerchProbeAdapter(probe=None, probe_path=None, model_path=None, labels=None, top_k=5)
#   classify(segment, sr)->list[Prediction]: Perch2 embed(1536) -> probe.predict_proba ->
#     ranked Predictions (names from probe.classes_, then labels arg, then species_<i>).
#     No probe -> [Prediction("embedding_only", 0.0)] + last_embedding stashed.
#   embed(waveform, sr)->np.ndarray[1536]   # delegates to Perch2Adapter (lazy TF).
#   Probe: explicit arg -> probe_path -> PERCH_V2_PROBE_PATH. Registered for
#   FAUN_CLASSIFIER=perch-probe; detections source SOURCE_PERCH_V2_PROBE = "model:perch-v2-probe".

# faun/settings: + perch_v2_probe_path (PERCH_V2_PROBE_PATH) — operator-supplied local probe.
# faun/detections: + SOURCE_PERCH_V2_PROBE = "model:perch-v2-probe".

# faun/cli (additive):
#   faun retrain --model now accepts yamnet|perch|perch-v2 (probe backbone; was yamnet-only).
#   faun export-labels --job <dir> --out <csv>   # detections.jsonl -> retrain CSV, keeping ONLY
#     is_ground_truth labels (human expert/ranger + confirmed/corrected); columns species,source,
#     status,segment_path,... consumable by `faun retrain`. Closes the review->retrain loop.

# deploy/Dockerfile.ml (NEW): TF Perch 2 serving image — same python:3.12-slim digest base as the
#   slim image + requirements-ml.txt (tensorflow-cpu==2.20.0 + kagglehub==1.0.2) + experiments/wrappers
#   (the Perch2Adapter._infer delegate). Defaults FAUN_CLASSIFIER=perch-v2 + PERCH_V2_MODEL_PATH=
#   /models/perch2 (model in a volume, not the image). slim deploy/Dockerfile stays TF-free = rollback.
```
