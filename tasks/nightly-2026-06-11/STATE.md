# Nightly 2026-06-11 — durable state

Plan: `/Users/user/.claude/plans/fancy-snacking-wadler.md` (v3). Orchestrator = main session.
On restart/compaction: re-read this file FIRST, continue from last checkpoint, do not redo done work.

## Legend
✅ done · 🔄 in progress · ⏭️ skipped (reason) · ❌ failed (reason)

## Phase 0 — Preflight
- ✅ gh auth: switched active → gianhu403-hash; it ALREADY has `workflow` scope (plan's "no-workflow" assumption outdated). Throwaway workflow-file push to `_authprobe_throwaway` SUCCEEDED → **push phases ON**. Branch deleted local+remote.
- ✅ Tag `v1-hackathon` created at 0ae49b4 (= demo pin) and pushed to origin (tag obj 319d9b3).
- ✅ deploy.yml → workflow_dispatch-only stub. Committed+pushed to main (903a2b8). GATE PASSED: no new deploy run (latest deploy run = 2026-05-08/0ae49b4). CI triggered on push (will be replaced by reorg).
- ✅ Worktree `../faun-wt/reorg` on branch `chore/reorg` created (at 903a2b8).
- ✅ STATE.md initialized. Scratch dir `/Users/user/sandbox/faun-nightly-artifacts/{research,cluster}` for non-repo artifacts (avoids reorg collision).
- ✅ REUSE-core verified: classifier.py lazy-imports TF (top-level only numpy/soundfile/os) → faun.ml.yamnet imports clean under requirements-pipeline. MODEL_PATH default v8→needs v7 fix. Actual weight: edge/audio/yamnet_forest_classifier_v7.keras.
- **PHASE 0 COMPLETE.**

## Phase 1 — launched concurrently (web ⊥ ssh ⊥ repo-worktree)
- ✅ Track A research DONE (5/5 surveys + synth; report at /Users/user/sandbox/faun-nightly-artifacts/research/research-report.md). KEY: top classifier = **Perch 2** (Apache 2.0, perch_v2_cpu via bioacoustics-model-zoo, needs Kaggle; fallback Perch 1 TFHub no-auth). BirdNET = **CC BY-NC-SA** (ShareAlike taints fine-tuned heads!) — inventory only. Detector = onset.py CPU stage-1 (+CLAP GPU verifier optional; NDSI unfit for <10s events). Without xeno-canto: only binary bird/no-bird (freefield1010/warblrb10k/PolandNFC). E1 ground: BirdNET inventory sanity (not species accuracy). CLAP weights CC0 no-auth.
- 🔄 Track B cluster: Workflow `wnnu2ccnb` (paths+creds → yadisk/datasets/images in tmux)
- 🔄 Track C reorg: Workflow `wxp03hm4t` (single Opus agent in ../faun-wt/reorg, self-verifies gates, commits chore/reorg). Main merges after.
- Awaiting completion notifications. Reorg gates merging of all Phase-2 waves.

## Phase 3 — ingress VERIFIED ✅ (done inline by main while Phase 1 runs)
- anchor nginx = `delphi-press-nginx-1` (nginx:1.27-alpine).
- faun.antopkin.ru vhost: `set $faun_upstream "100.64.0.1:8003"` (cluster) — CORRECT.
- GET https://faun.antopkin.ru/health → 200 `{"status":"ok"}`. Frozen demo live, served from cluster via anchor. (HEAD gives 405 — FastAPI /health is GET-only; not a fault.)
- deploy.yml already stubbed (Phase 0). vhost healthy → no nginx edit needed.
- DEFERRED to Phase 6: update docs/deployment.md + memory (server_delphi_press.md, MEMORY antopkin-vps line: VM deleted 2026-05-30, ingress on anchor).

## Cluster facts (verified this session)
- cluster-alex reachable: oleg, uid 1001, RTX 2060 SUPER 8192 MiB, /home 485G free.
- anchor reachable: deploy user.

## Phases
- Phase 0: 🔄 (almost done)
- Phase 1 A/B/C: pending
- Phase 2 W1-W5: pending
- Phase 3: pending
- Phase 4-8: pending

## Decisions / deviations log
- (none yet beyond gh-auth note above)
