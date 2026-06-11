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
