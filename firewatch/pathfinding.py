import heapq
from typing import Any

from .models import Camera, Event, Zone, ZoneLayout


DEFAULT_ROWS = 20
DEFAULT_COLS = 20


def _empty_cells(rows: int, cols: int) -> list[list[str]]:
    return [["empty" for _ in range(cols)] for _ in range(rows)]


def get_or_create_layout(zone: Zone) -> ZoneLayout:
    layout, _ = ZoneLayout.objects.get_or_create(
        zone=zone,
        defaults={"rows": DEFAULT_ROWS, "cols": DEFAULT_COLS, "cells_json": _empty_cells(DEFAULT_ROWS, DEFAULT_COLS)},
    )
    normalized = normalize_layout(layout.rows, layout.cols, layout.cells_json)
    if normalized != layout.cells_json:
        layout.cells_json = normalized
        layout.save(update_fields=["cells_json", "updated_at"])
    return layout


def normalize_layout(rows: int, cols: int, cells: Any) -> list[list[str]]:
    safe = _empty_cells(rows, cols)
    if not isinstance(cells, list):
        return safe

    valid_types = {"empty", "wall", "stairs", "entrance", "fire"}
    for r in range(min(rows, len(cells))):
        row = cells[r]
        if not isinstance(row, list):
            continue
        for c in range(min(cols, len(row))):
            cell = row[c]
            safe[r][c] = cell if cell in valid_types else "empty"
    return safe


def _camera_scores_for_fire(event: Event) -> dict[str, float]:
    scores: dict[str, float] = {}
    for frame in event.frame_detections.select_related("camera"):
        for det in frame.detections_json:
            if int(det.get("class_id", -1)) == 0:
                cam_id = frame.camera.camera_id
                conf = float(det.get("confidence", 0.0))
                scores[cam_id] = max(scores.get(cam_id, 0.0), conf)
    return scores


def choose_fire_camera(event: Event) -> Camera | None:
    scores = _camera_scores_for_fire(event)
    if scores:
        camera_id = max(scores, key=scores.get)
        return Camera.objects.filter(camera_id=camera_id).first()

    first_detection = event.frame_detections.select_related("camera").first()
    if first_detection:
        return first_detection.camera

    return event.zone.cameras.filter(active=True).first()


def _cell_cost(cell_type: str) -> float:
    if cell_type == "stairs":
        return 1.2
    return 1.0


def _neighbors(r: int, c: int, rows: int, cols: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc


def _build_walls(cells: list[list[str]]) -> set[tuple[int, int]]:
    walls = set()
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            if cell == "wall":
                walls.add((r, c))
    return walls


def _heuristic(node: tuple[int, int], goal: tuple[int, int]) -> float:
    # Manhattan distance is admissible for 4-neighbor grid with minimum move cost 1.0.
    return float(abs(node[0] - goal[0]) + abs(node[1] - goal[1]))


def astar_path(
    rows: int,
    cols: int,
    cells: list[list[str]],
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked_extra: set[tuple[int, int]] | None = None,
):
    blocked = _build_walls(cells)
    if blocked_extra:
        blocked |= blocked_extra

    if start in blocked or goal in blocked:
        return None

    g_score: dict[tuple[int, int], float] = {start: 0.0}
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    heap = [(_heuristic(start, goal), 0.0, start)]
    closed = set()

    while heap:
        _, cur_g, node = heapq.heappop(heap)
        if node in closed:
            continue
        closed.add(node)

        if node == goal:
            break

        for nxt in _neighbors(node[0], node[1], rows, cols):
            if nxt in blocked:
                continue
            weight = _cell_cost(cells[nxt[0]][nxt[1]])
            tentative_g = cur_g + weight
            if tentative_g < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative_g
                prev[nxt] = node
                f_score = tentative_g + _heuristic(nxt, goal)
                heapq.heappush(heap, (f_score, tentative_g, nxt))

    if goal not in g_score:
        return None

    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = prev[cur]
    path.append(start)
    path.reverse()

    return {"cost": round(g_score[goal], 3), "path": [[r, c] for r, c in path]}


def _entrances(cells: list[list[str]]) -> list[tuple[int, int]]:
    out = []
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            if cell == "entrance":
                out.append((r, c))
    return out


def _fire_cells(cells: list[list[str]]) -> list[tuple[int, int]]:
    out = []
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            if cell == "fire":
                out.append((r, c))
    return out


def compute_routes_for_event(event: Event, blocked_cells: list[list[int]] | None = None, max_alternatives: int = 3):
    layout = get_or_create_layout(event.zone)
    cells = normalize_layout(layout.rows, layout.cols, layout.cells_json)
    entrances = _entrances(cells)
    if not entrances:
        entrances = [(0, 0)]

    blocked_extra = {(int(p[0]), int(p[1])) for p in (blocked_cells or []) if isinstance(p, list) and len(p) == 2}

    fire_goals = _fire_cells(cells)
    target_type = "fire"
    target_camera_id = None

    if not fire_goals:
        camera = choose_fire_camera(event)
        if not camera:
            return {"error": "No fire cell or camera available for this event/zone."}
        camera_point = layout.camera_points_json.get(camera.camera_id)
        if not camera_point or len(camera_point) != 2:
            return {"error": f"No camera point configured in layout for {camera.camera_id}."}
        fire_goals = [(int(camera_point[0]), int(camera_point[1]))]
        target_type = "camera_fallback"
        target_camera_id = camera.camera_id

    best = None
    best_start = None
    goal = None
    for g in fire_goals:
        for ent in entrances:
            route = astar_path(layout.rows, layout.cols, cells, ent, g, blocked_extra=blocked_extra)
            if not route:
                continue
            if best is None or route["cost"] < best["cost"]:
                best = route
                best_start = ent
                goal = g

    if not best:
        return {
            "error": "No route found from any entrance to fire source.",
            "target_type": target_type,
            "target_camera_id": target_camera_id,
            "target_cell": [fire_goals[0][0], fire_goals[0][1]] if fire_goals else None,
        }

    primary_path = best["path"]
    alternatives = []
    seen = {tuple(tuple(cell) for cell in primary_path)}

    for node in primary_path[1:-1]:
        avoid = set(blocked_extra)
        avoid.add((node[0], node[1]))
        alt = astar_path(layout.rows, layout.cols, cells, best_start, goal, blocked_extra=avoid)
        if not alt:
            continue
        key = tuple(tuple(cell) for cell in alt["path"])
        if key in seen:
            continue
        seen.add(key)
        alternatives.append(alt)
        if len(alternatives) >= max_alternatives:
            break

    return {
        "event_id": event.event_id,
        "zone": event.zone.code,
        "target_type": target_type,
        "target_camera_id": target_camera_id,
        "target_cell": [goal[0], goal[1]],
        "entrance_cell": [best_start[0], best_start[1]] if best_start else None,
        "primary_route": best,
        "alternative_routes": alternatives,
        "blocked_cells": [[r, c] for r, c in blocked_extra],
        "layout": {
            "rows": layout.rows,
            "cols": layout.cols,
            "cells": cells,
            "camera_points": layout.camera_points_json,
        },
    }
