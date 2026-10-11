"""Delivery for the daily comparisons: ntfy push notifications (phone) and/or email.

Settings come from environment variables (GitHub Actions secrets in the daily run):
  NTFY_TOPIC                    push to https://ntfy.sh/<topic> (install the free ntfy app and subscribe to it)
  SMTP_HOST, SMTP_PORT,
  SMTP_USER, SMTP_PASSWORD,
  DIGEST_EMAIL_TO               send an email (e.g. with a Gmail app password)
"""

import os
import smtplib
from email.message import EmailMessage

import requests


def setting(name: str) -> str | None:
    return os.environ.get(name) or None


def channels() -> list[str]:
    out = []
    if setting("NTFY_TOPIC"):
        out.append("ntfy")
    if all(setting(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "DIGEST_EMAIL_TO")):
        out.append("email")
    return out


def send_ntfy(title: str, body: str, priority: int = 3, click: str | None = None) -> None:
    """Publish via ntfy's JSON API, which handles any Unicode in the title (headers can't)."""
    payload = {
        "topic": setting("NTFY_TOPIC"),
        "title": title,
        "message": body,
        "priority": priority,
        "tags": ["chart_with_upwards_trend"],
    }
    if click:
        payload["click"] = click
    resp = requests.post("https://ntfy.sh/", json=payload, timeout=15)
    resp.raise_for_status()


def send_email(title: str, body: str) -> None:
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = title, setting("SMTP_USER"), setting("DIGEST_EMAIL_TO")
    msg.set_content(body)
    with smtplib.SMTP(setting("SMTP_HOST"), int(setting("SMTP_PORT") or 587), timeout=20) as smtp:
        smtp.starttls()
        smtp.login(setting("SMTP_USER"), setting("SMTP_PASSWORD"))
        smtp.send_message(msg)


def send(title: str, body: str, priority: int = 3, click: str | None = None) -> list[str]:
    """Send through every configured channel. Returns the channels used."""
    used = channels()
    if "ntfy" in used:
        send_ntfy(title, body, priority, click)
    if "email" in used:
        send_email(title, body)
    return used
