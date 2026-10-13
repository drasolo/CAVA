"""The session clock: a small always-on-top window.

Started by the coach (pythonw, no console). It owns me/session.json while a
session runs: every second it advances the active clock, and it watches the
drill files so saving your code counts as activity. The clock turns green once
the session counts (20 active minutes by default). "stop" ends the session the
same way the coach does.
"""
import sys
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cava_practice import session, state  # noqa: E402

DRILLS = Path(__file__).resolve().parent.parent / "drills"
GREY, GREEN, AMBER = "#8b949e", "#3fb950", "#d29922"
BG, FG = "#0d1117", "#c9d1d9"


class Clock:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("cava")
        self.root.attributes("-topmost", True)
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        font = ("Consolas", 10)
        self.time = tk.Label(self.root, text="00:00", font=("Consolas", 22, "bold"), bg=BG, fg=GREY)
        self.what = tk.Label(self.root, text="", font=font, bg=BG, fg=FG)
        self.week = tk.Label(self.root, text="", font=font, bg=BG, fg=GREY)
        self.note = tk.Label(self.root, text="", font=font, bg=BG, fg=AMBER)
        stop = tk.Button(self.root, text="stop", font=font, command=self.stop, bg="#21262d", fg=FG,
                         activebackground="#30363d", relief="flat", padx=8)
        for w in (self.time, self.what, self.week, self.note):
            w.pack(padx=12, anchor="w")
        stop.pack(padx=12, pady=(2, 10), anchor="e")
        self.mtimes = self.scan()
        self.root.geometry("+40+40")
        self.root.after(200, self.update)

    def scan(self):
        return {p: p.stat().st_mtime for p in DRILLS.glob("d*.py")}

    def update(self):
        s = session.current()
        if not s:
            self.root.destroy()
            return
        now = self.scan()
        if now != self.mtimes:
            self.mtimes = now
            session.mark_activity()
        cfg = state.config()
        session.tick(s, state.now(), session.last_activity(), cfg["idle_minutes"] * 60)
        # Re-read before saving: the coach may have changed the drill or ended the session.
        fresh = session.current()
        if not fresh:
            self.root.destroy()
            return
        fresh.update({k: s[k] for k in ("active_seconds", "last_tick", "idle")})
        state.save(state.SESSION, fresh)

        secs = int(s.get("active_seconds", 0))
        counts = secs >= cfg["min_session_minutes"] * 60
        self.time.config(text=f"{secs // 3600}:{secs // 60 % 60:02d}:{secs % 60:02d}" if secs >= 3600
                         else f"{secs // 60:02d}:{secs % 60:02d}", fg=GREEN if counts else GREY)
        what = fresh.get("kind", "")
        if fresh.get("drill"):
            what = f"{fresh['drill']} · {what}"
        self.what.config(text=what + ("   (paused: idle)" if s.get("idle") else ""))
        wk = session.week_summary(state.load(state.SESSIONS, default=[]), state.today(),
                                  secs / 60 if counts else 0, fresh.get("kind", ""))
        self.week.config(text=f"week {session.fmt(wk['minutes'])} / {session.fmt(cfg['weekly_minutes'])}")
        pending = state.load(state.BACKUP).get("backup_pending")
        self.note.config(text="backup pending" if pending else "")
        self.root.after(1000, self.update)

    def stop(self):
        rec = session.finish()
        self.root.destroy()
        if rec:
            print(rec)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Clock().run()
