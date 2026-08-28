# Validation — P1 Linux 优化路线执行基础设施

## 1. Definition of done

- [ ] Linux user-level Python 3.10.19 prefix is installed without modifying shared Conda base;
- [ ] Torch 2.10.0+cu128 sees only the launcher-selected logical `cuda:0` and records the
  expected physical RTX 4090 identity;
- [ ] machine/environment/config/code manifests and hashes are complete and finite;
- [ ] occupied GPU, wrong GPU, CPU/RAM/disk shortage, dirty Git, missing command or malformed
  inventory all fail closed without process-control side effects;
- [ ] wait mode requires five consecutive eligible samples, times out after 48 hours, holds
  the project GPU lease, and never lowers thresholds;
- [ ] normal O1 Gate is baseline/H12 only; H4 is isolated diagnostic-only evidence;
- [ ] O1 repeat/window shards are atomic and explicit resume rejects mismatch/corruption;
- [ ] O1 Go receipt is verifiable by the future O2 runner; P1 never starts O2;
- [ ] Windows local development remains functional and every existing regression passes;
- [ ] owner Linux installation and short CUDA smoke evidence pass;
- [ ] P1 is reported only as infrastructure readiness, then the project proceeds to O1 Gate
  and, conditional on O1 Go, O2.

## 2. Owner installation verification

The owner runs commands from `/home/lzx/llm-a-mappo`; no `conda activate`, `sudo`, shared-base
package modification or credential is allowed. Expected results:

1. canonical interpreter reports Python 3.10.19;
2. Torch reports 2.10.0+cu128 and CUDA available;
3. constrained package report contains exact frozen versions;
4. editable project import succeeds;
5. freeze file and SHA-256 are written under the versioned P1 artifact directory.

## 3. Resource and safety verification

- [ ] fixture inventory maps physical index/name/memory/UUID/PCI deterministically;
- [ ] real server inventory records GPU 0 as RTX 4090 with at least 48000 MiB;
- [ ] the currently observed dual-GPU Python process makes a one-shot preflight fail;
- [ ] no implementation calls `kill`, `pkill`, `taskkill`, process suspend, renice or GPU reset;
- [ ] GPU lease contention fails before output/training side effects;
- [ ] a failed sample resets consecutive-free count to zero;
- [ ] timeout and interruption retain wait logs and return nonzero;
- [ ] disk/RAM/CPU thresholds and clean Git are literal manifest values.

## 4. O1 contract verification

- [ ] `gate --modes baseline h12` is the only normal accepted mode list;
- [ ] all 12 workers allocate model tensors on logical `cuda:0`;
- [ ] five repeats, 16/128 vector steps and 2+10 memory windows remain frozen;
- [ ] runtime ratio and persistent-memory formulas are byte-for-byte behaviorally unchanged;
- [ ] H4 requires a failed Gate summary and cannot contain or change `gate_pass`;
- [ ] every row includes physical GPU and environment provenance;
- [ ] Gate output cannot be reused after code/config/machine/environment hash drift.

## 5. Resume and O2 boundary verification

- [ ] valid completed shards are reused deterministically;
- [ ] corrupt, partial or mismatched shards fail closed rather than being silently replaced;
- [ ] only allowlisted infrastructure interruption is resumable;
- [ ] NaN, algorithm, safety and unknown failures remain non-resumable evidence;
- [ ] receipt exists only for a complete O1 Go and names the next required phase O2;
- [ ] no P1 code imports or invokes an O2 training entry point.

## 6. Commands to run locally

```powershell
& "D:\Anaconda3\envs\py310\python.exe" -m pytest `
  tests/test_linux_environment_contract.py `
  tests/test_linux_server_runtime.py `
  tests/test_o1_linux_gate_contract.py `
  tests/test_run_evidence.py `
  tests/test_run_o1_when_available.py `
  tests/test_reward_calibration_benchmark.py -q

& "D:\Anaconda3\envs\py310\python.exe" -m pytest -q
& "D:\Anaconda3\envs\py310\python.exe" -m flake8 rware llm_mappo eval train scripts figures/core
& "D:\Anaconda3\envs\py310\python.exe" -m build --wheel --no-isolation
& "D:\Anaconda3\envs\py310\python.exe" scripts/check_optimization_server.py --help
& "D:\Anaconda3\envs\py310\python.exe" scripts/run_o1_when_available.py --help
& "D:\Anaconda3\envs\py310\python.exe" scripts/benchmark_reward_calibration.py --help
git diff --check
git status --short
```

## 7. Owner-run Linux smoke

P1-G must generate the literal commands after implementation. P1-H requires the owner to run:

1. environment creation and verification;
2. one-shot occupied-resource No-Go evidence if GPU 0 is busy;
3. wait-mode eligibility evidence when GPU 0 becomes free;
4. one CUDA tensor/device smoke;
5. one 128-step optimization smoke on GPU 0;
6. artifact hash listing and Git status.

This smoke is short infrastructure validation, not O1 Gate or training evidence.

## 8. Merge criteria

- [ ] P1-A through P1-H complete;
- [ ] all local and owner Linux checks pass;
- [ ] Roadmap, TASKS, protocol, manifest, terminology and CHANGELOG agree;
- [ ] no secrets, credentials, checkpoints, large artifacts or server-specific mutable UUID
  are committed;
- [ ] working tree is clean and focused commits are reviewable;
- [ ] owner approves “P1 Linux optimization infrastructure ready” without claiming O1/O2;
- [ ] next action is owner-run O1 Gate; O1 Go makes O2 mandatory.
