# -*- coding: utf-8 -*-
# pip install ortools
from ortools.sat.python import cp_model
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time

# Gewichte/Strafen (Feinjustierung)
PENALTY_CLASS_MIX_PER_EXTRA_CLASS = 3     # je zusätzliche Klasse im Zimmer über 1
PENALTY_EMPTY_BED = 0                     # freie Betten bestrafen (0 = egal)
PENALTY_CROSS_GENDER_ROOM = 1000          # Sicherheitsnetz (soll 0 bleiben)
TEACHER_SHARED_ROOM_PENALTY = 2           # Strafe pro weiterer Lehrkraft im selben Zimmer (bevorzugt Einzelzimmer)
PENALTY_TEACHER_WRONG_CORRIDOR = 5        # Lehrkraft nicht auf Flur mit ihrer Klasse
# NEU: Klassen möglichst nicht über mehrere Flure splitten
PENALTY_CLASS_SPLIT_ACROSS_CORRIDORS = 4  # je zusätzlicher Flur pro Klasse über 1

# Solver-Parameter (Default – GUI kann überschreiben)
DEFAULT_MAX_TIME_SECONDS = 60.0
DEFAULT_NUM_WORKERS = 8
DEFAULT_PROGRESS_INTERVAL = 5.0


class ProgressPrinter(cp_model.CpSolverSolutionCallback):
    """ Gibt regelmäßig Fortschritt während der Optimierung aus. """
    def __init__(self, update_interval: float = 5.0):
        cp_model.CpSolverSolutionCallback.__init__(self)
        self._solution_count = 0
        self._start = time.time()
        self._last_print = self._start
        self._update_interval = update_interval

    def OnSolutionCallback(self):
        self._solution_count += 1
        now = time.time()
        if now - self._last_print >= self._update_interval:
            elapsed = now - self._start
            try:
                obj = self.ObjectiveValue()
                obj_str = f"{obj:.0f}" if isinstance(obj, (int, float)) else f"{obj}"
            except Exception:
                obj_str = "—"
            print(f"[{elapsed:6.1f}s] Lösungen: {self._solution_count:5d} | aktuelles Ziel: {obj_str}")
            self._last_print = now

    @property
    def solution_count(self) -> int:
        return self._solution_count


def solve_assignment(
    people: Dict[str, Dict],
    rooms: Dict[str, Dict],
    forbidden_pairs: List[Tuple[str, str]],
    corridors: List[str],
    required_teachers_per_corridor: Dict[str, List[str]],
    time_limit_s: float = DEFAULT_MAX_TIME_SECONDS,
    num_workers: int = DEFAULT_NUM_WORKERS,
    progress_interval: float = DEFAULT_PROGRESS_INTERVAL,
) -> Optional[Dict]:
    """
    people: {person_id: {name, gender("m"/"w"), role("student"/"teacher"), class_id|None, small_group_max|None}}
    rooms:  {room_id: {name, capacity:int, corridor:str}}
    forbidden_pairs: [(a_id, b_id), ...]
    corridors: ["A","B",...]
    required_teachers_per_corridor: {"A": ["t1","t2"], ...}

    Rückgabe:
      {
        "allocation": { room_id: [ {id,name,gender,role,class_id}, ... ], ... },
        "objective": float|None,
        "status": int,
        "stats": {...}
      }
    """
    model = cp_model.CpModel()

    persons = list(people.keys())
    room_ids = list(rooms.keys())

    genders = {"m", "w"}
    students = [p for p in persons if people[p]["role"] == "student"]
    teachers = [p for p in persons if people[p]["role"] == "teacher"]

    # Klassenliste nur aus Schüler:innen bilden (für Zimmermix & Flur-Tracker)
    classes = sorted({people[p]["class_id"] for p in students if people[p]["class_id"] is not None})

    # Entscheidungsvariablen
    x = {(p, r): model.NewBoolVar(f"x[{p},{r}]") for p in persons for r in room_ids}  # Person p in Zimmer r
    y = {(r, g): model.NewBoolVar(f"y[{r},{g}]") for r in room_ids for g in genders}  # Zimmer r hat Geschlecht g
    z = {(r, k): model.NewBoolVar(f"z[{r},{k}]") for r in room_ids for k in classes}  # Zimmer r hat Klasse k

    # Zusatzvariablen für Rollentrennung
    has_teacher = {r: model.NewBoolVar(f"has_teacher[{r}]") for r in room_ids}
    has_student = {r: model.NewBoolVar(f"has_student[{r}]") for r in room_ids}

    # 1) Jede Person genau einem Zimmer
    for p in persons:
        model.Add(sum(x[p, r] for r in room_ids) == 1)

    # 2) Kapazitäten
    for r in room_ids:
        model.Add(sum(x[p, r] for p in persons) <= rooms[r]["capacity"])

    # 3) Geschlechtertrennung pro Zimmer (für alle, inkl. Lehrkräfte)
    for r in room_ids:
        model.Add(sum(y[r, g] for g in genders) <= 1)  # höchstens ein Geschlecht
        for p in persons:
            g = people[p]["gender"]
            model.Add(x[p, r] <= y[r, g])

    # 4) Verbotene Paare
    for a, b in forbidden_pairs:
        if a in persons and b in persons:
            for r in room_ids:
                model.Add(x[a, r] + x[b, r] <= 1)

    # 5) "Kleine Gruppe"
    for p in persons:
        kmax = people[p].get("small_group_max")
        if kmax is not None:
            for r in room_ids:
                cap = rooms[r]["capacity"]
                model.Add(sum(x[q, r] for q in persons) - kmax <= (1 - x[p, r]) * cap)

    # 6) Pro Flur mind. eine Lehrkraft + (optional) konkret geforderte Lehrkräfte
    for c in corridors:
        rooms_on_c = [r for r in room_ids if rooms[r]["corridor"] == c]
        if teachers and rooms_on_c:
            model.Add(sum(x[t, r] for t in teachers for r in rooms_on_c) >= 1)
        for t in required_teachers_per_corridor.get(c, []):
            if t in teachers:
                model.Add(sum(x[t, r] for r in rooms_on_c) == 1)

    # 7) Klassenmix (weich) – z[r,k] wird 1, sobald jemand aus Klasse k in r liegt
    for r in room_ids:
        for k in classes:
            class_members = [p for p in students if people[p]["class_id"] == k]
            for p in class_members:
                model.Add(z[r, k] >= x[p, r])

    # 8) Rollentrennung hart: Lehrkräfte und Schüler:innen nie im selben Zimmer
    for r in room_ids:
        if teachers:
            for t in teachers:
                model.Add(has_teacher[r] >= x[t, r])
        if students:
            for s in students:
                model.Add(has_student[r] >= x[s, r])
        model.Add(has_teacher[r] + has_student[r] <= 1)

    # --- Flur-Hilfsvariablen ---
    corridor_rooms = {c: [r for r in room_ids if rooms[r]["corridor"] == c] for c in corridors}

    # class_on_c: Klasse k ist auf Flur c vertreten (mind. ein(e) Schüler:in aus k in einem Zimmer des Flurs)
    class_on_c = {}
    for c in corridors:
        for k in classes:
            v = model.NewBoolVar(f"class_on_c[{c},{k}]")
            class_on_c[(c, k)] = v
            rooms_on_c = corridor_rooms[c]
            members_k = [p for p in students if people[p]["class_id"] == k]
            for r in rooms_on_c:
                for p in members_k:
                    model.Add(v >= x[p, r])

    # teacher_on_c: Lehrkraft t liegt auf Flur c
    teacher_on_c = {}
    for t in teachers:
        for c in corridors:
            v = model.NewBoolVar(f"teacher_on_c[{t},{c}]")
            teacher_on_c[(t, c)] = v
            for r in corridor_rooms[c]:
                model.Add(v >= x[t, r])

    # mismatch: Lehrkraft mit Klassen-ID, aber auf Flur ohne diese Klasse
    mismatch = {}
    for t in teachers:
        k_t = people[t].get("class_id")
        if not k_t or k_t not in classes:
            continue
        for c in corridors:
            m = model.NewBoolVar(f"teacher_mismatch[{t},{c}]")
            mismatch[(t, c)] = m
            tc = teacher_on_c[(t, c)]
            ck = class_on_c[(c, k_t)]
            # m = 1 genau wenn tc==1 und ck==0
            model.Add(m <= tc)
            model.Add(m <= 1 - ck)
            model.Add(m >= tc - ck)

    # 9) Zielfunktion
    objective_terms = []

    # a) Zusätzliche Klassen im Zimmer bestrafen
    for r in room_ids:
        num_classes_in_r = sum(z[r, k] for k in classes) if classes else 0
        occupied_r = model.NewBoolVar(f"occupied[{r}]")
        model.Add(sum(x[p, r] for p in persons) >= 1).OnlyEnforceIf(occupied_r)
        model.Add(sum(x[p, r] for p in persons) == 0).OnlyEnforceIf(occupied_r.Not())

        extra_classes = model.NewIntVar(0, max(0, len(classes) - 1) if classes else 0, f"extra_classes[{r}]")
        if classes:
            model.Add(extra_classes >= num_classes_in_r - 1)
        model.Add(extra_classes <= (len(classes) if classes else 0) * occupied_r)
        if PENALTY_CLASS_MIX_PER_EXTRA_CLASS > 0 and classes:
            objective_terms.append(extra_classes * PENALTY_CLASS_MIX_PER_EXTRA_CLASS)

    # b) Freie Betten (optional)
    if PENALTY_EMPTY_BED > 0:
        for r in room_ids:
            empty_beds = rooms[r]["capacity"] - sum(x[p, r] for p in persons)
            objective_terms.append(empty_beds * PENALTY_EMPTY_BED)

    # c) Cross-Gender (Sicherheitsnetz)
    if PENALTY_CROSS_GENDER_ROOM > 0:
        for r in room_ids:
            both_gender = model.NewBoolVar(f"bad_gender_mix[{r}]")
            model.Add(y[r, "m"] + y[r, "w"] - both_gender <= 1)
            model.Add(both_gender <= y[r, "m"])
            model.Add(both_gender <= y[r, "w"])
            objective_terms.append(both_gender * PENALTY_CROSS_GENDER_ROOM)

    # d) Lehrkräfte bevorzugt im Einzelzimmer (weich)
    if TEACHER_SHARED_ROOM_PENALTY > 0 and teachers:
        for r in room_ids:
            num_teachers_in_r = sum(x[t, r] for t in teachers)
            extra_teachers = model.NewIntVar(0, len(teachers), f"extra_teachers[{r}]")
            model.Add(extra_teachers >= num_teachers_in_r - 1)
            objective_terms.append(extra_teachers * TEACHER_SHARED_ROOM_PENALTY)

    # e) Lehrkraft-Flur-Penalty (weiche Präferenz zum Klassenflur)
    if PENALTY_TEACHER_WRONG_CORRIDOR > 0:
        for (t, c), m in mismatch.items():
            objective_terms.append(m * PENALTY_TEACHER_WRONG_CORRIDOR)

    # f) NEU: Klassen möglichst nicht über mehrere Flure splitten
    #    Für jede Klasse k: extra_flure_k >= sum_c class_on_c[c,k] - 1
    if PENALTY_CLASS_SPLIT_ACROSS_CORRIDORS > 0 and corridors and classes:
        for k in classes:
            corridors_used = sum(class_on_c[(c, k)] for c in corridors)
            extra_flure = model.NewIntVar(0, max(0, len(corridors) - 1), f"class_extra_corridors[{k}]")
            model.Add(extra_flure >= corridors_used - 1)
            # Wenn Klasse gar nicht vertreten ist, ist corridors_used=0 -> extra_flure >= -1; das passt,
            # wir wollen dann aber keine Strafe. Begrenzen mit >=0 ist durch Domäne schon gegeben.
            objective_terms.append(extra_flure * PENALTY_CLASS_SPLIT_ACROSS_CORRIDORS)

    if objective_terms:
        model.Minimize(sum(objective_terms))
    else:
        model.Minimize(0)

    # Solver konfigurieren
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_workers

    progress_cb = ProgressPrinter(update_interval=progress_interval)
    print(f"Starte Optimierung… (Zeitlimit {time_limit_s:.0f}s, Threads {num_workers})")
    start = time.time()
    result_status = solver.Solve(model, progress_cb)
    elapsed = time.time() - start
    print(f"Suche beendet nach {elapsed:.1f}s. Gefundene Zwischenlösungen: {progress_cb.solution_count}")

    if result_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    # Lösung auslesen
    allocation_dd = defaultdict(list)
    for r in room_ids:
        for p in persons:
            if solver.Value(x[p, r]) == 1:
                info = people[p]
                allocation_dd[r].append({
                    "id": p,
                    "name": info.get("name", p),
                    "gender": info.get("gender"),
                    "role": info.get("role"),
                    "class_id": info.get("class_id"),
                })

    # In normales Dict wandeln (wichtig für Streamlit/JSON)
    allocation = {rid: list(occs) for rid, occs in allocation_dd.items()}

    try:
        obj_val = float(solver.ObjectiveValue())
    except Exception:
        obj_val = None

    return {
        "allocation": allocation,
        "objective": obj_val,
        "status": int(result_status),
        "stats": {
            "solve_time_s": elapsed,
            "solutions_seen": progress_cb.solution_count,
            "persons": len(persons),
            "rooms": len(room_ids),
        },
    }
