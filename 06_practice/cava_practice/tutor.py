"""Claude as a tutor, through the Claude Code CLI (`claude -p`), with no tools.

Rules it is given (and why): it never writes code, because the exam has no AI
and typing the code yourself is the point; it explains, asks, and points.
Phase 1 has the one-shot "explain differently"; the live chat comes later.
"""
import shutil
import subprocess

RULES = (
    "You are a tutor for a computer vision student practising for a lab exam with no AI allowed. "
    "Never write code, not even one line or a function name call with arguments. "
    "Explain ideas in plain words, use a small numeric example when it helps, and end with one "
    "short question that checks understanding. Keep it under 150 words."
)
NO_TOOLS = ["Bash", "Edit", "Write", "Read", "Glob", "Grep", "WebFetch", "WebSearch",
            "NotebookEdit", "Task", "Agent"]


def available() -> bool:
    return shutil.which("claude") is not None


def ask(prompt: str, timeout: int = 120) -> str:
    if not available():
        return "(Claude Code isn't installed on this machine, so the tutor is off.)"
    cmd = ["claude", "-p", prompt, "--append-system-prompt", RULES,
           "--disallowedTools", *NO_TOOLS]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                             timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return "(The tutor took too long. Try again, or open a Claude session yourself.)"
    return (out.stdout or out.stderr).strip()


def explain(name: str, written: str) -> None:
    print("\n    asking Claude for another angle ...")
    reply = ask(f"I didn't understand this explanation of '{name}':\n\n{written}\n\n"
                "Explain it differently, from a more basic starting point.")
    for line in reply.splitlines():
        print(f"      {line}")
