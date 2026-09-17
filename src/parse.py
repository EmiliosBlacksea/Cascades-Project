from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import re

@dataclass(frozen=True)
class RetweetEvent:
    leaf: int
    t: int
    path: List[int]  # includes root (prepended if missing)

_line_split = re.compile(r"\t+")

def parse_retweet_item(item: str) -> Optional[Tuple[List[int], int]]:
    """Parse one item: user1/user2/.../usern:retweet_time"""
    item = item.strip()
    if not item or ":" not in item:
        return None
    path_str, t_str = item.rsplit(":", 1)
    try:
        t = int(t_str)
    except ValueError:
        return None
    users: List[int] = []
    for u in path_str.split("/"):
        u = u.strip()
        if not u:
            continue
        try:
            users.append(int(u))
        except ValueError:
            return None
    if not users:
        return None
    return users, t

def parse_weibo_line(line: str):
    """
    Format:
      <message_id>\t<root_user_id>\t<publish_time>\t<retweet_number>\t<retweets>
    retweets: space-separated items.
    """
    parts = _line_split.split(line.rstrip("\n"))
    if len(parts) < 4:
        return None
    try:
        message_id = int(parts[0])
        root_user_id = int(parts[1])
        publish_time = int(parts[2])
        retweet_number_24h = int(parts[3])
    except ValueError:
        return None

    retweets_str = parts[4] if len(parts) >= 5 else ""
    items = retweets_str.strip().split(" ") if retweets_str.strip() else []
    events_raw: List[Tuple[List[int], int]] = []
    for it in items:
        parsed = parse_retweet_item(it)
        if parsed is None:
            continue
        events_raw.append(parsed)

    return {
        "message_id": message_id,
        "root_user_id": root_user_id,
        "publish_time": publish_time,
        "retweet_number_24h": retweet_number_24h,
        "events_raw": events_raw,
    }

# def build_incremental_events(root_user_id: int, events_raw: List[Tuple[List[int], int]]) -> List[RetweetEvent]:
#     """Convert raw events to chronological RetweetEvent; prepend root if missing."""
#     events: List[RetweetEvent] = []
#     for path_users, t in events_raw:
#         if not path_users:
#             continue
#         if path_users[0] != root_user_id:
#             path = [root_user_id] + path_users
#         else:
#             path = path_users
#         events.append(RetweetEvent(leaf=path[-1], t=t, path=path))
#     events.sort(key=lambda e: e.t)
#     return events

def build_incremental_events(root_user_id: int, events_raw: List[Tuple[List[int], int]]) -> List[RetweetEvent]:
    """Convert raw events to chronological RetweetEvent; prepend root if missing.
    Drop the synthetic root record 'root:0' (path=[root], t=0).
    """
    events: List[RetweetEvent] = []
    for path_users, t in events_raw:
        if not path_users:
            continue

        # normalize: ensure root at front
        path = path_users if path_users[0] == root_user_id else [root_user_id] + path_users

        # drop synthetic root event like "128:0"
        if len(path) == 1 and path[0] == root_user_id and t == 0:
            continue

        events.append(RetweetEvent(leaf=path[-1], t=t, path=path))

    events.sort(key=lambda e: e.t)
    return events
