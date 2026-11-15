"""Chat defaults shared by browser and OpenAI-compatible coding clients."""

import json


def apply_defaults(data):
    if not isinstance(data, dict):
        return data
    if data.get("max_tokens") is None and data.get("max_completion_tokens") is None:
        data["max_tokens"] = 16384
    budget = data.get("thinking_token_budget")
    if budget is None or budget == -1:
        data["thinking_token_budget"] = 8192
    elif type(budget) is int and budget > 8192:
        data["thinking_token_budget"] = 8192
    data.setdefault("reasoning_effort", "medium")
    kwargs = data.get("chat_template_kwargs")
    if kwargs is None:
        kwargs = {}
        data["chat_template_kwargs"] = kwargs
    if isinstance(kwargs, dict):
        kwargs.setdefault("enable_thinking", data["reasoning_effort"] != "none")
        kwargs.setdefault("preserve_thinking", True)
    return data


class ChatDefaults:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/v1/chat/completions"
        ):
            return await self.app(scope, receive, send)
        chunks = []
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            chunks.append(event.get("body", b""))
            if not event.get("more_body", False):
                break
        body = b"".join(chunks)
        try:
            data = json.loads(body)
            body = json.dumps(apply_defaults(data)).encode()
        except (ValueError, UnicodeError):
            pass  # The normal API validator reports malformed input.
        scope = dict(scope)
        scope["headers"] = [
            (key, value)
            for key, value in scope.get("headers", [])
            if key.lower() != b"content-length"
        ]
        scope["headers"].append((b"content-length", str(len(body)).encode()))
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)
