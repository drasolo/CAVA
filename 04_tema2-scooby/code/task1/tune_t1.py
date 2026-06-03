from __future__ import annotations

import os
import re
import sys
import csv
import time
import json
import queue
import shutil
import signal
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# --- paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_T1 = PROJECT_ROOT / "cod" / "task1" / "run_t1.py"
EVAL = PROJECT_ROOT / "evaluare" / "cod_evaluare" / "evalueaza_solutie.py"

DEBUG_ROOT = PROJECT_ROOT / "debug" / "t1_tune"
JOBS_DIR = DEBUG_ROOT / "jobs"
STATUS_DIR = DEBUG_ROOT / "status"
RESULTS_CSV = DEBUG_ROOT / "results.csv"
BEST_ENV = DEBUG_ROOT / "best.env"

DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)
STATUS_DIR.mkdir(parents=True, exist_ok=True)


# Add near the top (with the other regexes)
IMG_RE = re.compile(r"^\[OK\]\s+\[(\d+)\]\s+(.+?):\s+det_pre_nms=", re.M)

def tail_text(path: Path, max_bytes: int = 16_384) -> str:
    """
    Read only the last `max_bytes` of a text file (fast, avoids loading huge logs).
    """
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes), os.SEEK_SET)
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""

def last_image_progress(job_dir: Path) -> tuple[int | None, str | None]:
    """
    Extract last scanned image index + rel filename from job's run_t1 stdout.
    Returns (img_idx, rel_path) like (29, 'validare/0029.jpg') or (None, None).
    """
    p = job_dir / "run_t1.stdout.txt"
    txt = tail_text(p)
    if not txt:
        return None, None
    matches = list(IMG_RE.finditer(txt))
    if not matches:
        return None, None
    m = matches[-1]
    return int(m.group(1)), m.group(2).strip()




def _load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)

_load_env_file(PROJECT_ROOT / ".env")


AP_RE = re.compile(r"\[Task1\]\s+Average Precision \(AP\)\s*=\s*([0-9.]+)")

# ---------------------------
# Job definition
# ---------------------------
@dataclass(frozen=True)
class Job:
    job_id: int
    STEP: int
    PYRAMID_SCALES: str
    DETECTION_SCORE_THRESH: float
    TOPK_PRE_NMS: int
    NMS_IOU_THRESH: float

    def env_patch(self) -> dict[str, str]:
        return {
            "RUN_SPLIT": "val",
            "STEP": str(self.STEP),
            "PYRAMID_SCALES": self.PYRAMID_SCALES,
            "DETECTION_SCORE_THRESH": str(self.DETECTION_SCORE_THRESH),
            "TOPK_PRE_NMS": str(self.TOPK_PRE_NMS),
            "NMS_IOU_THRESH": str(self.NMS_IOU_THRESH),
            # critical: avoid overwrites between workers
            "TASK1_RUN_SUBDIR": f"task1_runs/job_{self.job_id:04d}",
        }


# ---------------------------
# Helpers
# ---------------------------
def now_s() -> float:
    return time.time()


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    payload = json.dumps(data, indent=2)
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())

    # os.replace is atomic on Windows for same filesystem
    os.replace(str(tmp), str(path))



def run_cmd(
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_s: int,
) -> tuple[int, float]:
    t0 = now_s()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        try:
            p = subprocess.run(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=out,
                stderr=err,
                text=True,
                timeout=timeout_s,
            )
            return int(p.returncode), now_s() - t0
        except subprocess.TimeoutExpired:
            err.write(f"\n[TUNE] TIMEOUT after {timeout_s}s\n")
            return 124, now_s() - t0

def parse_ap_from_eval_stdout(stdout_text: str) -> float | None:
    m = AP_RE.search(stdout_text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def read_text_safe(p: Path) -> str:
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def load_dotenv_student_folder() -> str | None:
    # Ensure project .env is loaded for this process too.
    _load_env_file(PROJECT_ROOT / ".env")

    v = os.getenv("STUDENT_FOLDER", "").strip()
    return v or None

def ensure_csv_header(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "job_id","status","ap",
            "step","pyramid_scales","score_thr","topk_pre_nms","nms_iou",
            "run_s","eval_s","detections","matched","duplicates","error"
        ])

def append_csv(row: list[Any]) -> None:
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(row)

def parse_eval_stats(stdout_text: str) -> tuple[int | None, int | None, int | None]:
    # expects: [Task1] detections=583  GT=332  matched=198  duplicates=1
    m = re.search(r"\[Task1\]\s+detections=(\d+)\s+GT=(\d+)\s+matched=(\d+)\s+duplicates=(\d+)", stdout_text)
    if not m:
        return None, None, None
    return int(m.group(1)), int(m.group(3)), int(m.group(4))

# ---------------------------
# Worker
# ---------------------------



def worker_loop(worker_id: int, jobs_q: "queue.Queue[Job]", results_q: "queue.Queue[dict]") -> None:
    py = sys.executable

    eval_dir = PROJECT_ROOT / "evaluare" / "cod_evaluare"
    gt_task1 = PROJECT_ROOT / "validare" / "task1_gt_validare.txt"

    IMG_LINE_RE = re.compile(r"^\[OK\]\s+\[(\d+)\]\s+(.+?\.jpg)\b", re.IGNORECASE)

    def update_img_progress(run_stdout_path: Path, status_path: Path, total_imgs: int | None) -> None:
        if not run_stdout_path.exists():
            return
        try:
            with run_stdout_path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                sz = f.tell()
                f.seek(max(0, sz - 16384), os.SEEK_SET)
                chunk = f.read().decode("utf-8", errors="replace")
        except Exception:
            return

        last_i = None
        last_name = None
        for line in reversed(chunk.splitlines()):
            m = IMG_LINE_RE.search(line.strip())
            if m:
                last_i = int(m.group(1))
                last_name = m.group(2)
                break

        if last_i is None:
            return

        try:
            s = json.loads(status_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            s = {"worker": worker_id}

        s["img_i"] = last_i
        if total_imgs is not None:
            s["img_total"] = total_imgs
        s["img_name"] = last_name
        atomic_write_json(status_path, s)

    while True:
        try:
            job = jobs_q.get_nowait()
        except queue.Empty:
            return

        job_dir = JOBS_DIR / f"job_{job.job_id:04d}"
        job_dir.mkdir(parents=True, exist_ok=True)

        status_path = STATUS_DIR / f"worker_{worker_id}.json"

        # isolate each job via STUDENT_FOLDER
        base_student = (os.getenv("STUDENT_FOLDER", "").strip() or "STUDENT")
        job_student = f"{base_student}_TUNE_{job.job_id:04d}"

        env = dict(os.environ)
        env["STUDENT_FOLDER"] = job_student
        env.update(job.env_patch())

        # flush stdout fast
        env["PYTHONUNBUFFERED"] = "1"

        # CRITICAL: outputs are under <STUDENT_FOLDER>/<TASK1_RUN_SUBDIR>/
        run_subdir = env.get("TASK1_RUN_SUBDIR", "task1").replace("\\", "/").strip("/")
        solution_task1_dir = PROJECT_ROOT / "evaluare" / "fisiere_solutie" / job_student / Path(run_subdir)

        (job_dir / "params.json").write_text(json.dumps(asdict(job), indent=2), encoding="utf-8")
        (job_dir / "env_patch.json").write_text(json.dumps(job.env_patch(), indent=2), encoding="utf-8")
        (job_dir / "student_folder.txt").write_text(job_student, encoding="utf-8")
        (job_dir / "solution_task1_dir.txt").write_text(str(solution_task1_dir), encoding="utf-8")

        atomic_write_json(status_path, {
            "worker": worker_id,
            "job_id": job.job_id,
            "stage": "RUN_T1",
            "started": now_s(),
            "student_folder": job_student,
            "task1_run_subdir": run_subdir,
            "img_i": 0,
            "img_total": None,
            "img_name": None,
        })

        run_stdout = job_dir / "run_t1.stdout.txt"
        run_stderr = job_dir / "run_t1.stderr.txt"

        timeout_s = int(os.getenv("TUNE_RUN_TIMEOUT_S", "5400"))
        t0 = now_s()

        with run_stdout.open("w", encoding="utf-8", buffering=1) as out, \
             run_stderr.open("w", encoding="utf-8", buffering=1) as err:
            try:
                p = subprocess.Popen(
                    [py, str(RUN_T1)],
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=out,
                    stderr=err,
                    text=True,
                )

                last_poll = 0.0
                while True:
                    rc = p.poll()

                    if now_s() - t0 > timeout_s:
                        p.kill()
                        err.write(f"\n[TUNE] TIMEOUT after {timeout_s}s\n")
                        rc_run = 124
                        break

                    if now_s() - last_poll > 2.0:
                        update_img_progress(run_stdout, status_path, total_imgs=None)
                        last_poll = now_s()

                    if rc is not None:
                        rc_run = int(rc)
                        break

                    time.sleep(0.2)

                dt_run = now_s() - t0

            except Exception as ex:
                err.write(f"\n[TUNE] EXCEPTION: {ex}\n")
                rc_run = 1
                dt_run = now_s() - t0

        if rc_run != 0:
            err_msg = f"run_t1 failed rc={rc_run}"
            results_q.put({
                "job": job, "status": "fail", "ap": None,
                "run_s": dt_run, "eval_s": 0.0,
                "detections": None, "matched": None, "duplicates": None,
                "error": err_msg, "stdout_eval": "",
            })
            atomic_write_json(status_path, {"worker": worker_id, "job_id": job.job_id, "stage": "DONE", "status": "fail", "error": err_msg})
            continue

        det_path = solution_task1_dir / "detections_all_faces.npy"
        scr_path = solution_task1_dir / "scores_all_faces.npy"
        fn_path  = solution_task1_dir / "file_names_all_faces.npy"

        if not (det_path.exists() and scr_path.exists() and fn_path.exists()):
            err_msg = (
                "Missing Task1 .npy outputs in expected folder.\n"
                f"Expected:\n  {det_path}\n  {scr_path}\n  {fn_path}\n"
                "Check STUDENT_FOLDER / TASK1_RUN_SUBDIR wiring."
            )
            results_q.put({
                "job": job, "status": "fail", "ap": None,
                "run_s": dt_run, "eval_s": 0.0,
                "detections": None, "matched": None, "duplicates": None,
                "error": err_msg, "stdout_eval": "",
            })
            atomic_write_json(status_path, {"worker": worker_id, "job_id": job.job_id, "stage": "DONE", "status": "fail", "error": "missing npy outputs"})
            continue

        atomic_write_json(status_path, {"worker": worker_id, "job_id": job.job_id, "stage": "EVAL_T1", "run_s": dt_run})

        eval_stdout = job_dir / "eval_t1.stdout.txt"
        eval_stderr = job_dir / "eval_t1.stderr.txt"

        solution_path_arg = str(solution_task1_dir).replace("\\", "/") + "/"
        gt_path_arg = str(gt_task1).replace("\\", "/")

        eval_code = (
            "import os, sys\n"
            "sys.path.insert(0, os.getcwd())\n"
            "from evalueaza_solutie import evaluate_results_task1\n"
            f"evaluate_results_task1('{solution_path_arg}', '{gt_path_arg}', True)\n"
        )

        rc_eval, dt_eval = run_cmd(
            cmd=[py, "-c", eval_code],
            cwd=eval_dir,
            env=env,
            stdout_path=eval_stdout,
            stderr_path=eval_stderr,
            timeout_s=int(os.getenv("TUNE_EVAL_TIMEOUT_S", "1800")),
        )

        out_text = read_text_safe(eval_stdout)
        ap = parse_ap_from_eval_stdout(out_text)
        dets, matched, dup = parse_eval_stats(out_text)

        status = "ok" if (rc_eval == 0 and ap is not None) else "fail"
        err_msg = "" if status == "ok" else f"eval_t1 failed rc={rc_eval} ap={ap}"

        results_q.put({
            "job": job,
            "status": status,
            "ap": ap,
            "run_s": dt_run,
            "eval_s": dt_eval,
            "detections": dets,
            "matched": matched,
            "duplicates": dup,
            "error": err_msg,
            "stdout_eval": out_text,
        })

        atomic_write_json(status_path, {"worker": worker_id, "job_id": job.job_id, "stage": "DONE", "status": status, "ap": ap, "run_s": dt_run, "eval_s": dt_eval})



# --------------------------------
# Main dashboard + orchestration
# --------------------------------
# Replace your print_dashboard() with this version
def print_dashboard(total: int, done: int, best_ap: float, best_job: Job | None) -> None:
    lines: list[str] = []
    header = f"[TUNE] progress {done}/{total} | best_ap={best_ap:.3f}"
    if best_job:
        header += f" | best_job={best_job.job_id:04d}"
    lines.append(header)

    for p in sorted(STATUS_DIR.glob("worker_*.json")):
        try:
            s = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            # status might be mid-atomic replace on Windows; skip this refresh
            continue

        wid = s.get("worker")
        jid = s.get("job_id")
        stage = s.get("stage")
        st = s.get("status", "")
        ap = s.get("ap", None)

        msg = f"  worker {wid}: job={jid} stage={stage}"
        if st:
            msg += f" status={st}"
        if ap is not None:
            try:
                msg += f" ap={float(ap):.3f}"
            except Exception:
                pass

        if stage == "RUN_T1":
            img_i = s.get("img_i")
            img_total = s.get("img_total")
            img_name = s.get("img_name")
            if img_i is not None:
                msg += f" | img={img_i}/{img_total or '?'}"
                if img_name:
                    msg += f" {img_name}"

        lines.append(msg)

    sys.stdout.write("\n" + "\n".join(lines) + "\n")
    sys.stdout.flush()

def write_best_env(job: Job) -> None:
    """
    Writes a runnable env file for a *final* (non-tuning) run.

    - Forces TASK1_RUN_SUBDIR=task1 so evaluator reads the canonical folder.
    - Strips any _TUNE_XXXX suffix if tuner is running with per-job STUDENT_FOLDER.
    """
    student = os.getenv("STUDENT_FOLDER", "").strip()
    if not student:
        student = "STUDENT"

    # If we're running tuning jobs, STUDENT_FOLDER is often like:
    # 341_Solomon_Dragos_TUNE_0039
    # We want the base folder for final submission:
    student_base = re.sub(r"_TUNE_\d{4}$", "", student)

    env_lines = [
        f"STUDENT_FOLDER={student_base}",
        "TASK1_RUN_SUBDIR=task1",
        "",
        "RUN_SPLIT=test",
        "FILENAME_MODE=basename",
        "",
        f"STEP={job.STEP}",
        f"PYRAMID_SCALES={job.PYRAMID_SCALES}",
        "",
        f"DETECTION_SCORE_THRESH={job.DETECTION_SCORE_THRESH}",
        f"TOPK_PRE_NMS={job.TOPK_PRE_NMS}",
        f"NMS_IOU_THRESH={job.NMS_IOU_THRESH}",
        "",
    ]
    BEST_ENV.write_text("\n".join(env_lines), encoding="utf-8")



def build_grid() -> list[Job]:
    # You can expand this grid safely; keep it sane
    steps = [6, 8, 10]
    pyramids = [
        "1.0,0.9,0.8,0.7,0.6",
        "1.0,0.85,0.7,0.6",
    ]
    score_thrs = [1.6, 1.8, 2.0, 2.2]
    topks = [60, 80, 120]
    nms_thrs = [0.10, 0.15]

    jobs: list[Job] = []
    jid = 0
    for st in steps:
        for py in pyramids:
            for thr in score_thrs:
                for tk in topks:
                    for nm in nms_thrs:
                        jobs.append(Job(
                            job_id=jid,
                            STEP=st,
                            PYRAMID_SCALES=py,
                            DETECTION_SCORE_THRESH=thr,
                            TOPK_PRE_NMS=tk,
                            NMS_IOU_THRESH=nm,
                        ))
                        jid += 1
    return jobs

def main() -> None:
    # must be set (your .env won’t be auto-loaded by this script)
    if not load_dotenv_student_folder():
        raise RuntimeError("Set STUDENT_FOLDER in your shell env (or run in a context where it's set).")

    workers = int(os.getenv("TUNE_WORKERS", "4"))

    jobs = build_grid()
    total = len(jobs)

    print(f"[TUNE] grid_size={total} workers={workers}")
    print(f"[TUNE] python={sys.executable}")
    print(f"[TUNE] run_t1={RUN_T1}")

    ensure_csv_header(RESULTS_CSV)

    jobs_q: "queue.Queue[Job]" = queue.Queue()
    results_q: "queue.Queue[dict]" = queue.Queue()

    for j in jobs:
        jobs_q.put(j)

    import threading
    threads = []
    for wid in range(workers):
        t = threading.Thread(target=worker_loop, args=(wid, jobs_q, results_q), daemon=True)
        t.start()
        threads.append(t)

    done = 0
    best_ap = -1.0
    best_job: Job | None = None

    # consume results + refresh dashboard
    last_dash = 0.0
    try:
        while done < total:
            try:
                r = results_q.get(timeout=0.5)
            except queue.Empty:
                r = None

            if r is not None:
                job: Job = r["job"]
                ap = r["ap"]

                append_csv([...])

                if ap is not None and ap > best_ap:
                    best_ap = ap
                    best_job = job
                    write_best_env(job)

                done += 1

            if now_s() - last_dash > 1.5:
                print_dashboard(total, done, best_ap if best_ap >= 0 else 0.0, best_job)
                last_dash = now_s()

    except KeyboardInterrupt:
        print("\n[TUNE] interrupted by user")

    finally:
        print(f"[TUNE] partial results: {RESULTS_CSV}")
        if best_job:
            print(f"[TUNE] best so far AP={best_ap:.3f} job={best_job.job_id:04d}")
            print(f"[TUNE] best env so far: {BEST_ENV}")


    for t in threads:
        t.join(timeout=0.1)

    print(f"[TUNE] done. results: {RESULTS_CSV}")
    if best_job:
        print(f"[TUNE] best AP={best_ap:.3f} job={best_job.job_id:04d}")
        print(f"[TUNE] best env written to: {BEST_ENV}")

if __name__ == "__main__":
    main()
