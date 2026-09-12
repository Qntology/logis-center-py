import json
import os
import socket
import sys
import urllib.request
from typing import List, Optional

DEFAULT_PORT = 9222


def is_port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def find_free_port(start: int = DEFAULT_PORT, tries: int = 20) -> int:
    for i in range(tries):
        p = start + i
        if not is_port_open(p):
            return p
    return start


def enable_remote_debugging(port: int = DEFAULT_PORT) -> dict:
    port = int(port)

    existing = os.environ.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS", "")
    flags = [
        f"--remote-debugging-port={port}",
        "--remote-allow-origins=*",
    ]
    parts = [existing] if existing else []
    for f in flags:
        if f.split("=")[0] not in existing:
            parts.append(f)
    merged = " ".join(parts).strip()

    os.environ["WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS"] = merged
    os.environ["NMS_DEVTOOLS_PORT"] = str(port)
    os.environ.setdefault("PYWEBVIEW_LOG", "info")

    if sys.platform.startswith("linux"):
        os.environ.setdefault("WEBKIT_DISABLE_COMPOSITING_MODE", "1")
        os.environ["WEBKIT_INSPECTOR_SERVER"] = f"127.0.0.1:{port}"

    if sys.platform == "darwin":
        os.environ["PYWEBVIEW_GUI"] = os.environ.get("PYWEBVIEW_GUI", "")

    return {
        "port": port,
        "url": f"http://127.0.0.1:{port}",
        "flags": merged,
        "platform": sys.platform,
    }


def webview_capabilities() -> dict:
    caps = {
        "installed": False,
        "version": "",
        "gui": "",
        "toggle_devtools": False,
        "evaluate_js": False,
        "create_file_dialog": False,
    }
    try:
        import webview
    except Exception:
        return caps

    caps["installed"] = True
    caps["version"] = str(getattr(webview, "__version__", "") or "")

    try:
        gui = getattr(webview, "guilib", None)
        caps["gui"] = str(getattr(gui, "renderer", "") or getattr(gui, "__name__", "") or "")
    except Exception:
        pass

    try:
        win_cls = getattr(webview, "Window", None)
        if win_cls is not None:
            caps["toggle_devtools"] = hasattr(win_cls, "toggle_devtools")
            caps["evaluate_js"] = hasattr(win_cls, "evaluate_js")
            caps["create_file_dialog"] = hasattr(win_cls, "create_file_dialog")
    except Exception:
        pass

    if getattr(webview, "windows", None):
        w = webview.windows[0]
        caps["toggle_devtools"] = hasattr(w, "toggle_devtools")
        caps["evaluate_js"] = hasattr(w, "evaluate_js")

    return caps


def try_native_devtools() -> dict:
    try:
        import webview
    except Exception as e:
        return {"ok": False, "error": f"pywebview 임포트 실패: {e}"}

    if not getattr(webview, "windows", None):
        return {"ok": False, "error": "활성 창이 없습니다."}

    win = webview.windows[0]

    if hasattr(win, "toggle_devtools"):
        try:
            win.toggle_devtools()
            return {"ok": True, "mode": "native"}
        except Exception as e:
            return {"ok": False, "error": f"toggle_devtools 실패: {e}"}

    ver = str(getattr(webview, "__version__", "?"))
    return {
        "ok": False,
        "error": (
            f"이 pywebview 버전({ver})에는 toggle_devtools 가 없습니다. "
            f"pywebview>=5.0 으로 올리거나 원격 디버깅을 사용하세요."
        ),
    }


def list_targets(port: int = DEFAULT_PORT, timeout: float = 1.0) -> List[dict]:
    url = f"http://127.0.0.1:{port}/json/list"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return data


def devtools_url(port: int = DEFAULT_PORT) -> str:
    for t in list_targets(port):
        u = t.get("devtoolsFrontendUrl")
        if u:
            if u.startswith("/"):
                return f"http://127.0.0.1:{port}{u}"
            return u
    return f"http://127.0.0.1:{port}"


def open_in_browser(port: int = DEFAULT_PORT) -> dict:
    import webbrowser

    if not is_port_open(port):
        return {
            "ok": False,
            "error": (
                f"디버깅 포트 {port} 가 열려 있지 않습니다.\n"
                f"--devtools 옵션으로 앱을 실행했는지 확인하세요."
            ),
        }

    url = devtools_url(port)
    try:
        webbrowser.open(url)
        return {"ok": True, "url": url, "port": port}
    except Exception as e:
        return {"ok": False, "error": str(e), "url": url}


def status(port: int = DEFAULT_PORT) -> dict:
    open_ = is_port_open(port)
    targets = list_targets(port) if open_ else []
    caps = webview_capabilities()

    modes = []
    if caps.get("toggle_devtools"):
        modes.append("native")
    if open_:
        modes.append("remote")
    modes.append("inapp")

    return {
        "enabled": bool(os.environ.get("WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS")),
        "port": port,
        "listening": open_,
        "targets": len(targets),
        "url": f"http://127.0.0.1:{port}",
        "devtools_url": devtools_url(port) if open_ else "",
        "platform": sys.platform,
        "webview": caps,
        "modes": modes,
        "preferred": modes[0],
        "hint": _hint(open_, caps),
    }


def _hint(listening: bool, caps: dict) -> str:
    if caps.get("toggle_devtools"):
        return "F12 로 내장 DevTools 를 열 수 있습니다."
    if listening:
        return "크롬 주소창에 http://127.0.0.1:%s 를 입력하세요." % os.environ.get(
            "NMS_DEVTOOLS_PORT", DEFAULT_PORT
        )
    ver = caps.get("version") or "?"
    return (
        f"pywebview {ver} 에는 내장 DevTools 가 없고 원격 포트도 닫혀 있습니다.\n"
        f"앱을 --devtools 옵션으로 재시작하거나, 인앱 디버그 패널(F12)을 사용하세요.\n"
        f"내장 DevTools 를 쓰려면: pip install -U \"pywebview>=5.0\""
    )