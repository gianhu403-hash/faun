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
