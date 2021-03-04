import os
import shutil
import time
import typing


def wait_for(
    callback: typing.Callable,
    msg: typing.Optional[str] = None,
    exit_msg: typing.Optional[str] = None
):
    line_start = ""
    while True:
        value = callback()
        if value:
            break
        if msg is None:
            continue

        for i in range(3):
            dots = "." * (i + 1)
            spaces = " " * (2 - i)
            print(msg + dots + spaces, end="\r", flush=True)
            time.sleep(0.5)
        line_start = "\n"

    exit_msg = exit_msg or ""
    print(f"{line_start}{exit_msg}")
    return value


def clear_repo(repo_dir):
    for d in next(os.walk(repo_dir))[1]:
        print(f"Removing model {d}")
        shutil.rmtree(os.path.join(repo_dir, d))
