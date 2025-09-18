# -*- coding: utf-8 -*-
# pip install streamlit pandas ortools
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple
from solver import solve_assignment

st.set_page_config(page_title="Zimmerverteilung – Editor & Optimierer", layout="wide")
st.title("🏨 Zimmerverteilung – Editor & Optimierer")

# ---------- Defaults ----------
DEFAULT_PEOPLE = pd.DataFrame([
    {"id":"s1","name":"Ali","gender":"m","role":"student","class_id":"7a","small_group_max":None},
    {"id":"s2","name":"Bilal","gender":"m","role":"student","class_id":"7a","small_group_max":None},
    {"id":"s3","name":"Cem","gender":"m","role":"student","class_id":"7b","small_group_max":None},
    {"id":"s4","name":"Dina","gender":"w","role":"student","class_id":"7a","small_group_max":3},
    {"id":"s5","name":"Elif","gender":"w","role":"student","class_id":"7b","small_group_max":None},
    {"id":"t1","name":"Herr Roth","gender":"m","role":"teacher","class_id":None,"small_group_max":None},
    {"id":"t2","name":"Frau Blau","gender":"w","role":"teacher","class_id":None,"small_group_max":None},
])

DEFAULT_ROOMS = pd.DataFrame([
    {"id":"r101","name":"Sternschnuppe","capacity":4,"corridor":"A"},
    {"id":"r102","name":"Mondlicht","capacity":3,"corridor":"A"},
    {"id":"r201","name":"Sonnenaufgang","capacity":4,"corridor":"B"},
    {"id":"r202","name":"Nordwind","capacity":2,"corridor":"B"},
])

DEFAULT_FORBIDDEN = pd.DataFrame([
    {"a":"s1","b":"s3"},
])

DEFAULT_REQ = pd.DataFrame([
    {"corridor":"A","teacher_id":"t2"},
    {"corridor":"B","teacher_id":"t1"},
])

# ---------- Session State ----------
if "people" not in st.session_state:
    st.session_state.people = DEFAULT_PEOPLE.copy()
if "rooms" not in st.session_state:
    st.session_state.rooms = DEFAULT_ROOMS.copy()
if "forbidden_pairs" not in st.session_state:
    st.session_state.forbidden_pairs = DEFAULT_FORBIDDEN.copy()
if "required_teachers_per_corridor" not in st.session_state:
    st.session_state.required_teachers_per_corridor = DEFAULT_REQ.copy()

# ---------- Sidebar Options ----------
with st.sidebar:
    st.header("Solver-Optionen")
    time_limit = st.slider("Zeitlimit (Sek.)", 5, 300, 60, 5)
    workers = st.slider("Threads", 1, 16, 8)
    progress_every = st.slider("Fortschrittsintervall (Sek.)", 1, 30, 5)
    st.markdown("---")
    show_export = st.checkbox("JSON/CSV-Export-Buttons anzeigen", value=True)

# ---------- Snapshot vor dem Formular (stabile Optionslisten) ----------
people_snapshot = st.session_state.people.copy()
rooms_snapshot  = st.session_state.rooms.copy()

people_ids_snapshot   = people_snapshot["id"].dropna().astype(str).tolist()
teacher_ids_snapshot  = people_snapshot.query("role=='teacher'")["id"].astype(str).tolist()
corridors_snapshot    = sorted(set(rooms_snapshot["corridor"].dropna().astype(str).tolist())) or ["A","B","C"]

# ---------- Editoren im Formular ----------
with st.form("editors", clear_on_submit=False):
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("👥 Personen (editierbar)")
        st.caption("Spalten: id, name, gender{m/w}, role{student/teacher}, class_id(bei Lehrkräften leer), small_group_max")
        edited_people = st.data_editor(
            people_snapshot,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "gender": st.column_config.SelectboxColumn(options=["m","w"]),
                "role": st.column_config.SelectboxColumn(options=["student","teacher"]),
                "small_group_max": st.column_config.NumberColumn(step=1, min_value=0, format="%d"),
            },
            key="people_editor",
        )
    with c2:
        st.subheader("🚪 Zimmer (editierbar)")
        st.caption("Spalten: id, name, capacity, corridor")
        edited_rooms = st.data_editor(
            rooms_snapshot,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "capacity": st.column_config.NumberColumn(step=1, min_value=1, format="%d"),
            },
            key="rooms_editor",
        )

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("⛔️ Verbotene Paare (editierbar)")
        st.caption("Spalten: a, b → dürfen nicht im selben Zimmer sein")
        edited_forbidden = st.data_editor(
            st.session_state.forbidden_pairs,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "a": st.column_config.SelectboxColumn(options=people_ids_snapshot),
                "b": st.column_config.SelectboxColumn(options=people_ids_snapshot),
            },
            key="forbidden_editor",
        )
    with c4:
        st.subheader("🧑‍🏫 Lehrkräfte pro Gang (editierbar)")
        st.caption("Spalten: corridor, teacher_id → diese Lehrkraft MUSS in diesem Gang liegen")
        edited_req = st.data_editor(
            st.session_state.required_teachers_per_corridor,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "corridor": st.column_config.SelectboxColumn(options=corridors_snapshot),
                "teacher_id": st.column_config.SelectboxColumn(options=teacher_ids_snapshot),
            },
            key="req_editor",
        )

    submitted = st.form_submit_button("Änderungen übernehmen", type="secondary")

# Nur wenn geklickt wurde, Session-State aktualisieren
if submitted:
    st.session_state.people = edited_people.reset_index(drop=True)
    st.session_state.rooms = edited_rooms.reset_index(drop=True)
    st.session_state.forbidden_pairs = edited_forbidden.reset_index(drop=True)
    st.session_state.required_teachers_per_corridor = edited_req.reset_index(drop=True)
    st.success("Änderungen übernommen ✅")

# ---------- Helpers ----------
def normalize_nullable(v):
    if pd.isna(v) or v == "":
        return None
    return v

def df_to_people(df: pd.DataFrame) -> Dict:
    out = {}
    for _, row in df.iterrows():
        pid = str(row.get("id","")).strip()
        if not pid:
            continue
        role = str(row.get("role","")).strip()
        gender = str(row.get("gender","")).strip()
        out[pid] = {
            "name": str(row.get("name","")).strip() or pid,
            "gender": gender if gender in {"m","w"} else "m",
            "role": role if role in {"student","teacher"} else "student",
            "class_id": normalize_nullable(row.get("class_id")),
            "small_group_max": None if pd.isna(row.get("small_group_max")) or row.get("small_group_max")=="" else int(row.get("small_group_max")),
        }
    return out

def df_to_rooms(df: pd.DataFrame) -> Dict:
    out = {}
    for _, row in df.iterrows():
        rid = str(row.get("id","")).strip()
        if not rid:
            continue
        name = str(row.get("name","")).strip() or rid
        corridor = str(row.get("corridor","")).strip() or "A"
        cap = row.get("capacity", 1)
        try:
            cap = int(cap)
        except Exception:
            cap = 1
        out[rid] = {"name": name, "capacity": max(1, cap), "corridor": corridor}
    return out

def df_to_forbidden(df: pd.DataFrame) -> List[Tuple[str,str]]:
    pairs = []
    for _, row in df.iterrows():
        a = str(row.get("a","")).strip()
        b = str(row.get("b","")).strip()
        if a and b and a != b:
            pairs.append((a,b))
    return pairs

def df_to_required(df: pd.DataFrame) -> Dict[str, List[str]]:
    req = {}
    for _, row in df.iterrows():
        c = str(row.get("corridor","")).strip()
        t = str(row.get("teacher_id","")).strip()
        if c and t:
            req.setdefault(c, []).append(t)
    return req

# ---------- Run ----------
st.markdown("---")
run = st.button("🧠 Optimierung starten", type="primary")

if run:
    people = df_to_people(st.session_state.people)
    rooms = df_to_rooms(st.session_state.rooms)
    forbidden_pairs = df_to_forbidden(st.session_state.forbidden_pairs)
    required_teachers_per_corridor = df_to_required(st.session_state.required_teachers_per_corridor)

    if not people:
        st.error("Keine Personen definiert.")
    elif not rooms:
        st.error("Keine Zimmer definiert.")
    else:
        with st.spinner("Optimiere…"):
            corridors = sorted({v["corridor"] for v in rooms.values()})
            result = solve_assignment(
                people=people,
                rooms=rooms,
                forbidden_pairs=forbidden_pairs,
                corridors=corridors,
                required_teachers_per_corridor=required_teachers_per_corridor,
                time_limit_s=time_limit,
                num_workers=workers,
                progress_interval=progress_every,
            )

        if not result:
            st.error("Keine zulässige Lösung gefunden.")
        else:
            st.success("Fertig!")
            alloc = result.get("allocation", {})

            if isinstance(alloc, list) or not isinstance(alloc, dict):
                st.error("Allocation-Format unerwartet. Bitte 'solver.py' aus diesem Projekt verwenden.")
                st.stop()

            # Ergebnis-Tabelle
            rows = []
            for rid, occs in alloc.items():
                if rid not in rooms:
                    continue
                for o in occs:
                    rows.append({
                        "room_id": rid,
                        "room_name": rooms[rid]["name"],
                        "corridor": rooms[rid]["corridor"],
                        "capacity": rooms[rid]["capacity"],
                        "person_id": o["id"],
                        "name": o["name"],
                        "role": o["role"],
                        "gender": o["gender"],
                        "class_id": o["class_id"],
                    })

            out_df = pd.DataFrame(rows).sort_values(["corridor","room_name","role","gender","name"]) if rows else pd.DataFrame(
                columns=["room_id","room_name","corridor","capacity","person_id","name","role","gender","class_id"]
            )

            st.subheader("Ergebnis")
            st.dataframe(out_df, use_container_width=True)

            cdl, cinfo = st.columns([1,1])
            with cdl:
                csv = out_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Ergebnis als CSV", data=csv, file_name="room_assignment.csv", mime="text/csv")
                if show_export:
                    st.download_button("📥 people.json",
                        st.session_state.people.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
                        "people.json", "application/json")
                    st.download_button("📥 rooms.json",
                        st.session_state.rooms.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
                        "rooms.json", "application/json")
            with cinfo:
                st.markdown("**Solver-Infos**")
                st.json({
                    "objective": result.get("objective"),
                    "status": result.get("status"),
                    "stats": result.get("stats"),
                })

# Hinweise
st.markdown("""
**Tipps**
- Änderungen an Tabellen erst mit **„Änderungen übernehmen“** speichern (verhindert Rerun-Verluste).
- IDs müssen eindeutig sein (`id`-Spalte).
- Lehrkräfte (`role=teacher`) werden **nie** mit Schüler:innen gemischt; bevorzugt Einzelzimmer.
- Zimmername `name` ist optional (fällt sonst auf die `id` zurück).
""")
