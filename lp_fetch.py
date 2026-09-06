"""Читалка страниц через Lightpanda CDP. Usage: lp_fetch.py URL"""
import json
import sys

import requests
import websocket

CDP_HTTP = "http://127.0.0.1:9333"


def fetch(url: str, wait_ms: int = 4000) -> str:
    # Lightpanda: вкладка создаётся PUT'ом на /json/new (PUT-запрос, не POST)
    r = requests.put(f"{CDP_HTTP}/json/new?{url}")
    target = r.json() if r.text.strip().startswith("{") else json.loads(
        requests.get(f"{CDP_HTTP}/json/list").json() and "{}" or "{}")
    ws_url = target["webSocketDebuggerUrl"]
    ws = websocket.create_connection(ws_url, timeout=30)
    ws.send(json.dumps({"id": 1, "method": "Runtime.enable"}))
    ws.recv()
    ws.send(json.dumps({"id": 2, "method": "Runtime.evaluate", "params": {
        "expression": f"await new Promise(r => setTimeout(r, {wait_ms})); "
                      "document.body ? document.body.innerText : ''",
        "awaitPromise": True, "returnByValue": True,
    }}))
    while True:
        msg = json.loads(ws.recv())
        if msg.get("id") == 2:
            result = msg.get("result", {})
            value = result.get("result", {}).get("value", "")
            break
    ws.close()
    requests.get(f"{CDP_HTTP}/json/close/{target['id']}")
    return value


if __name__ == "__main__":
    import sys
    import requests  # noqa: E402
    print(fetch(sys.argv[1]))
else:
    import requests  # noqa: E402