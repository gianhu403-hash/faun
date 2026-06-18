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
