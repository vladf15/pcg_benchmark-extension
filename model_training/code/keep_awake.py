"""Keep Windows from going to sleep while a long job runs.

    python keep_awake.py -- python train.py --steps 30000000 ...   (run a command)
    python keep_awake.py --pid 12345                               (watch a running process)

Every `--interval` minutes (default 20) it calls
SetThreadExecutionState(ES_SYSTEM_REQUIRED), which resets the system idle
timer once, the same thing the timer sees when the machine is used.  It stops
as soon as the command or process ends, and passes the command's exit code
through.

What it leaves alone: no power plan or registry setting is changed, nothing is
typed or clicked, and the display is not held on (ES_DISPLAY_REQUIRED is not
set), so the screen still turns off and locks on its own schedule.  Sleep the
user asks for (lid, power button, Start menu) still happens.

Why a periodic reset rather than ES_CONTINUOUS: a continuous request stays in
force until the process that set it exits or clears it, so a crash in the
middle of a job could leave it set for that process's lifetime.  A one-shot
reset every 20 minutes lapses by itself under any Windows sleep timeout longer
than 20 minutes (the default on mains power is 30).
"""
import argparse
import ctypes
import subprocess
import sys
import time

ES_SYSTEM_REQUIRED = 0x00000001
STILL_ACTIVE = 259
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


def _poke():
    """Reset the system idle timer once.  Returns False off Windows."""
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.kernel32.SetThreadExecutionState(ES_SYSTEM_REQUIRED))


def _pid_alive(pid):
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--interval", type=float, default=20.0, help="minutes between resets")
    parser.add_argument("--pid", type=int, help="watch this running process instead of starting a command")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="command to run, after --")
    args = parser.parse_args()
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if (args.pid is None) == (not cmd):
        parser.error("give either --pid or a command after --")
    if sys.platform != "win32":
        print("keep_awake: not on Windows, nothing to do", file=sys.stderr)
    every = max(1.0, args.interval * 60.0)

    if cmd:
        proc = subprocess.Popen(cmd)
        while True:
            _poke()
            try:
                return proc.wait(timeout=every)
            except subprocess.TimeoutExpired:
                continue
            except KeyboardInterrupt:
                # Ctrl+C reaches the child too (same console); wait for it.
                return proc.wait()

    last = 0.0
    while _pid_alive(args.pid):
        if time.monotonic() - last >= every:
            _poke()
            last = time.monotonic()
        time.sleep(min(60.0, every))
    return 0


if __name__ == "__main__":
    sys.exit(main())
