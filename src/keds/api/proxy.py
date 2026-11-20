"""Explicit public scheme for deployments with TLS terminated by a proxy."""
from __future__ import annotations

from starlette.responses import Response


class GradioUploadProgressMiddleware:
    """Close Gradio 5's orphaned progress stream without reporting upload success."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        # Gradio can reset the ID before the browser opens the progress stream
        # after a fast upload (gradio-app/gradio#12002).
        if (scope["type"] == "http" and scope["method"] == "GET"
                and scope["path"] == "/ui/gradio_api/upload_progress"
                and scope["query_string"] == b"upload_id=undefined"):
            return await Response(": no upload to track\n\n", media_type="text/event-stream")(scope, receive, send)
        return await self.app(scope, receive, send)


class PublicSchemeMiddleware:
    """Restore the configured public scheme before Gradio builds absolute URLs.

    Enable only when the deployment's external protocol is known. This does not
    trust client-provided forwarding headers. Local development leaves it off.
    """

    def __init__(self, app, scheme):
        if scheme not in ("http", "https"):
            raise ValueError("Public scheme must be http or https")
        self.app = app
        self.scheme = scheme

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            scope = dict(scope)
            scope["scheme"] = ("wss" if self.scheme == "https" else "ws") if scope["type"] == "websocket" else self.scheme
            # Gradio consults this header directly when x-forwarded-host exists.
            scope["headers"] = [(key, value) for key, value in scope.get("headers", [])
                                if key.lower() != b"x-forwarded-proto"]
            scope["headers"].append((b"x-forwarded-proto", self.scheme.encode("ascii")))
        await self.app(scope, receive, send)
