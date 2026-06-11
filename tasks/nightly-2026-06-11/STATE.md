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
- ✅ Track C reorg DONE: commit a0ec4d1 on chore/reorg, all 5 gates green (verified independently by main: 44 passed, imports OK, grep rc=1, 0 deletions, 269≥254 files). MERGED to main `521b9b7` (--no-ff), pushed, **CI SUCCESS**. Note: edge keras v7 was tracked → moved to legacy/edge/audio/ (correct). Business docs were untracked → handled by main (below).
- ✅ Business docs committed by main on main `2502323`: new_context/new_info/ФАВН.pdf/cloud/Расчеты.pdf → docs/business/{smeta,meetings,hackathon,strategy}; tasks/ working docs + .mcp.json (no secrets) tracked; .claude local state gitignored. Working tree CLEAN.
- ✅ Track B cluster DONE: paths created; **creds НЕТ (kaggle/hf/xeno-canto)**; yadisk скрипт качает (манифест 1655 файлов; реальная структура: **A1 A2 A3 A4 + RECORDER + аудиоловушки-офис** — офисная папка = denoise-референс; НЕ A1..A5); datasets: ESC-50 + DCASE (ff1010 5.8GB + warblr 4.9GB c archive.org) качаются, FSC22/UrbanSound8K/Watkins SKIP (нет чистых no-auth ссылок); **оба образа собраны, гейты PASS**: faun-ml-torch (torch 2.12.0+cu130, cuda_available=True на RTX 2060 SUPER), faun-ml-cpu (tensorflow-cpu 2.21, birdnetlib 0.18.1). Гоча: uv pip вис на TCP-ретрансмитах даже с --network host → переведено на plain pip (IPv4 single-stream); Dockerfiles в faun-data/images/.

## Phase 2 — code waves launched (4 worktrees from main 2502323, disjoint paths)
- ✅ W1 core DONE: 144 passed in worktree (verified by main), committed 66a3c59, **MERGED to main 4aa2039**, pushed; CI in_progress (check before next merge).
- ✅ W2 apiui DONE: 54 passed + api smoke in worktree; UI 4 browser iterations (screenshots in faun-nightly-artifacts/ui-iter-*.png; subagent used local Chromium — Playwright MCP unavailable in its context). Committed 5a4c0a1, **MERGED 5fc7549**, CI SUCCESS. jobs/ gitignored.
- ✅ W3 **MERGED 6443700** after W2 (order kept); merged-main suite 177 passed locally.
- ✅ **Phase-5 e2e smokes done EARLY by main on merged main:** CLI smoke caught real seam drift (api.run_pipeline vs W1 CsvWriter/AudioFileEntry APIs) → fixed in api.py by main (commit "fix(api): align run_pipeline with real W1 module APIs", pushed). CLI: synthetic 48k trap dir → CSV (A1, 2.23s, StubAdapter preds) ✅. API e2e with real chain: POST→done(progress 1.0)→CSV→UI 200 ✅. Suite 177 passed.
- ✅ W3 adapters DONE: 67 passed, lazy-import gate verified (no TF/birdnetlib pulled on import); committed+pushed feat/classifier-adapters. HOLD merge until W2 lands (order core→apiui→adapters). NB: birdnetlib/Perch API signatures are assumed (mocked tests) — live integration check is in W5/E-experiments.
- ✅ W4 expcode DONE: runner+wrappers+exp_e0..e10, 8 smoke tests; code rsynced to cluster:/home/oleg/faun-data/code/ (smoke_ok). Committed ba3b1db, **MERGED f8c2561** (resolved .gitignore conflict W2 jobs/ vs W4 experiments/), suite 177 passed, pushed. **ALL FOUR WAVES MERGED.** Stub-tested only — real numbers from cluster run.
- ✅ Phase 6 docs DONE (README/pipeline.md/deployment.md rewritten, CLAUDE.md proofread, mkdocs nav v2+Legacy) — committed+pushed.
- ✅ Phase 6 memory DONE: new project_pilot_bioacoustics, reference_cluster_alex_faun, feedback_fable_safety_routing, feedback_forgejo_github_only; server_delphi_press.md → DECOMMISSIONED; MEMORY.md полностью переписан (v2 focus, legacy section).
- ✅ Phase 5 final audit DONE (Fable code-reviewer): **SHIP-WITH-CHANGES, 0 blockers, 3 majors** → все 3 пофикшены main'ом + yamnet timeout minor: atomic manifest write, honest multi-trap sidecar, tests/pipeline/test_e2e.py (единственный немокнутый прогон run_pipeline). Suite **180 passed**, pushed. Отчёт: faun-nightly-artifacts/final-audit.md. Отложено на июль: onset frame_index семантика, UI poll-on-error, унификация api job-store с faun.jobs, CSV-парсер в UI (split на запятой).
- Cluster live status 08:55: yadisk 500/1655 files 56G zero-fail; datasets dcase_bad downloading (1.5G total); images STILL BUILDING; **creds all NO (kaggle/hf/xeno-canto)** → E10 skips, Perch via TFHub v1 no-auth.
- W5 experiments-run: waits for Track B images+datasets AND W4 rsync.
- Merge order (Phase 5): core→apiui→adapters→exp, pytest in worktree before each, green CI after each.

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
