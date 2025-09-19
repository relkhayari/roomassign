# -*- coding: utf-8 -*-
# Requirements:
#   pip install streamlit pandas ortools reportlab
import os, json
from io import BytesIO
from datetime import date
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import mm
from reportlab.lib import colors

from solver import solve_assignment  # keep solver.py in same folder

st.set_page_config(page_title="Zimmerverteilung – Editor & Optimierer", layout="wide")
st.title("🏨 Zimmerverteilung – Editor & Optimierer")

# ------------------------------ Paths / Persistence ------------------------------
DATA_DIR = "data"
PEOPLE_PATH = os.path.join(DATA_DIR, "people.json")
ROOMS_PATH = os.path.join(DATA_DIR, "rooms.json")
FORBIDDEN_PATH = os.path.join(DATA_DIR, "forbidden_pairs.json")
REQ_PATH = os.path.join(DATA_DIR, "required_teachers_per_corridor.json")
os.makedirs(DATA_DIR, exist_ok=True)

def load_df(path: str, default_df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(path):
        return default_df.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except Exception as e:
        st.warning(f"Konnte {path} nicht laden ({e}); nutze Defaults.")
        return default_df.copy()

def save_df(df: pd.DataFrame, path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(json.loads(df.to_json(orient="records", force_ascii=False)), f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"Speichern fehlgeschlagen ({path}): {e}")

# Normalize whatever the editor stored into a pandas DataFrame (handles Streamlit quirks)
def _editor_to_df(key: str) -> pd.DataFrame:
    val = st.session_state.get(key, None)
    if isinstance(val, pd.DataFrame):
        return val
    if isinstance(val, list):
        try:
            return pd.DataFrame(val)
        except Exception:
            return pd.DataFrame()
    if isinstance(val, dict):
        try:
            return pd.DataFrame(val)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()

# ------------------------------ Defaults ------------------------------
DEFAULT_PEOPLE = pd.DataFrame([
    {"id":"s1","name":"Ali","gender":"m","role":"student","class_id":"7a","small_group_max":None},
    {"id":"s2","name":"Bilal","gender":"m","role":"student","class_id":"7a","small_group_max":None},
    {"id":"s3","name":"Cem","gender":"m","role":"student","class_id":"7b","small_group_max":None},
    {"id":"s4","name":"Dina","gender":"w","role":"student","class_id":"7a","small_group_max":3},
    {"id":"s5","name":"Elif","gender":"w","role":"student","class_id":"7b","small_group_max":None},
    {"id":"t1","name":"Herr Roth","gender":"m","role":"teacher","class_id":"7b","small_group_max":None},
    {"id":"t2","name":"Frau Blau","gender":"w","role":"teacher","class_id":"7a","small_group_max":None},
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

# ------------------------------ Init session_state from disk ------------------------------
if "people" not in st.session_state:
    st.session_state.people = load_df(PEOPLE_PATH, DEFAULT_PEOPLE)
if "rooms" not in st.session_state:
    st.session_state.rooms = load_df(ROOMS_PATH, DEFAULT_ROOMS)
if "forbidden_pairs" not in st.session_state:
    st.session_state.forbidden_pairs = load_df(FORBIDDEN_PATH, DEFAULT_FORBIDDEN)
if "required_teachers_per_corridor" not in st.session_state:
    st.session_state.required_teachers_per_corridor = load_df(REQ_PATH, DEFAULT_REQ)

# Cache last solver result so downloads don’t clear the view
if "last_result" not in st.session_state:
    st.session_state.last_result = None  # dict with alloc, out_df, pdf_bytes, csv_bytes, meta

# ------------------------------ Sidebar ------------------------------
with st.sidebar:
    st.header("Solver-Optionen")
    time_limit = st.slider("Zeitlimit (Sek.)", 5, 300, 60, 5)
    workers = st.slider("Threads", 1, 16, 8)
    progress_every = st.slider("Fortschrittsintervall (Sek.)", 1, 30, 5)
    st.markdown("---")
    st.header("Daten")
    if st.button("💾 Manuell speichern"):
        save_df(st.session_state.people, PEOPLE_PATH)
        save_df(st.session_state.rooms, ROOMS_PATH)
        save_df(st.session_state.forbidden_pairs, FORBIDDEN_PATH)
        save_df(st.session_state.required_teachers_per_corridor, REQ_PATH)
        st.success("Gespeichert.")
    if st.button("♻️ Auf Defaults zurücksetzen"):
        st.session_state.people = DEFAULT_PEOPLE.copy()
        st.session_state.rooms = DEFAULT_ROOMS.copy()
        st.session_state.forbidden_pairs = DEFAULT_FORBIDDEN.copy()
        st.session_state.required_teachers_per_corridor = DEFAULT_REQ.copy()
        save_df(st.session_state.people, PEOPLE_PATH)
        save_df(st.session_state.rooms, ROOMS_PATH)
        save_df(st.session_state.forbidden_pairs, FORBIDDEN_PATH)
        save_df(st.session_state.required_teachers_per_corridor, REQ_PATH)
        st.success("Zurückgesetzt & gespeichert.")
    show_export = st.checkbox("JSON/CSV-Export-Buttons anzeigen", value=True)

# ------------------------------ CSV Uploaders (per table) ------------------------------
st.markdown("### 📤 CSV-Uploads (optional)")
u1, u2, u3, u4 = st.columns(4)

def _apply_people_csv(df: pd.DataFrame):
    cols = ["id","name","gender","role","class_id","small_group_max"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    st.session_state.people = df.reset_index(drop=True)
    save_df(st.session_state.people, PEOPLE_PATH)
    st.success("People CSV geladen & gespeichert.")

def _apply_rooms_csv(df: pd.DataFrame):
    cols = ["id","name","capacity","corridor"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df["capacity"] = pd.to_numeric(df["capacity"], errors="coerce").fillna(1).astype(int)
    df["corridor"] = df["corridor"].fillna("A").astype(str)
    df = df[cols]
    st.session_state.rooms = df.reset_index(drop=True)
    save_df(st.session_state.rooms, ROOMS_PATH)
    st.success("Rooms CSV geladen & gespeichert.")

def _apply_forbidden_csv(df: pd.DataFrame):
    cols = ["a","b"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    st.session_state.forbidden_pairs = df.reset_index(drop=True)
    save_df(st.session_state.forbidden_pairs, FORBIDDEN_PATH)
    st.success("Forbidden Pairs CSV geladen & gespeichert.")

def _apply_req_csv(df: pd.DataFrame):
    cols = ["corridor","teacher_id"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    st.session_state.required_teachers_per_corridor = df.reset_index(drop=True)
    save_df(st.session_state.required_teachers_per_corridor, REQ_PATH)
    st.success("Required Teachers per Corridor CSV geladen & gespeichert.")

with u1:
    up = st.file_uploader("👥 people.csv", type=["csv"], key="people_csv_upl")
    if up is not None:
        try:
            df = pd.read_csv(up)
            _apply_people_csv(df)
        except Exception as e:
            st.error(f"Fehler beim Laden people.csv: {e}")

with u2:
    up = st.file_uploader("🚪 rooms.csv", type=["csv"], key="rooms_csv_upl")
    if up is not None:
        try:
            df = pd.read_csv(up)
            _apply_rooms_csv(df)
        except Exception as e:
            st.error(f"Fehler beim Laden rooms.csv: {e}")

with u3:
    up = st.file_uploader("⛔ forbidden_pairs.csv", type=["csv"], key="forbidden_csv_upl")
    if up is not None:
        try:
            df = pd.read_csv(up)
            _apply_forbidden_csv(df)
        except Exception as e:
            st.error(f"Fehler beim Laden forbidden_pairs.csv: {e}")

with u4:
    up = st.file_uploader("🧑‍🏫 required_teachers_per_corridor.csv", type=["csv"], key="req_csv_upl")
    if up is not None:
        try:
            df = pd.read_csv(up)
            _apply_req_csv(df)
        except Exception as e:
            st.error(f"Fehler beim Laden required_teachers_per_corridor.csv: {e}")

st.markdown("---")

# ------------------------------ Snapshots for stable option lists ------------------------------
people_snapshot = st.session_state.people.copy()
rooms_snapshot  = st.session_state.rooms.copy()
people_ids_snapshot   = people_snapshot["id"].dropna().astype(str).tolist()
teacher_ids_snapshot  = people_snapshot.query("role=='teacher'")["id"].astype(str).tolist() if "role" in people_snapshot.columns else []
corridors_snapshot    = sorted(set(rooms_snapshot["corridor"].dropna().astype(str).tolist())) or ["A","B","C"]

# ------------------------------ Auto-save callbacks (per table) ------------------------------
def save_people_callback():
    df = _editor_to_df("people_editor")
    st.session_state.people = df.reset_index(drop=True)
    save_df(st.session_state.people, PEOPLE_PATH)
    st.toast("👥 Personen gespeichert", icon="💾")

def save_rooms_callback():
    df = _editor_to_df("rooms_editor")
    st.session_state.rooms = df.reset_index(drop=True)
    if "capacity" in st.session_state.rooms.columns:
        st.session_state.rooms["capacity"] = pd.to_numeric(
            st.session_state.rooms["capacity"], errors="coerce"
        ).fillna(1).astype(int)
    if "corridor" in st.session_state.rooms.columns:
        st.session_state.rooms["corridor"] = st.session_state.rooms["corridor"].fillna("A").astype(str)
    save_df(st.session_state.rooms, ROOMS_PATH)
    st.toast("🚪 Zimmer gespeichert", icon="💾")

def save_forbidden_callback():
    df = _editor_to_df("forbidden_editor")
    st.session_state.forbidden_pairs = df.reset_index(drop=True)
    save_df(st.session_state.forbidden_pairs, FORBIDDEN_PATH)
    st.toast("⛔️ Verbote gespeichert", icon="💾")

def save_req_callback():
    df = _editor_to_df("req_editor")
    st.session_state.required_teachers_per_corridor = df.reset_index(drop=True)
    save_df(st.session_state.required_teachers_per_corridor, REQ_PATH)
    st.toast("🧑‍🏫 Gang-Zuordnung gespeichert", icon="💾")

# ------------------------------ Editable tables (auto-save) ------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("👥 Personen (auto-save)")
    st.caption("Enter/Tab speichert sofort.")
    st.data_editor(
        st.session_state.people,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "gender": st.column_config.SelectboxColumn(options=["m","w"]),
            "role": st.column_config.SelectboxColumn(options=["student","teacher"]),
            "small_group_max": st.column_config.NumberColumn(step=1, min_value=0, format="%d"),
        },
        key="people_editor",
        on_change=save_people_callback,
    )

with c2:
    st.subheader("🚪 Zimmer (auto-save)")
    st.caption("Enter/Tab speichert sofort.")
    st.data_editor(
        st.session_state.rooms,
        num_rows="dynamic",
        use_container_width=True,
        column_config={"capacity": st.column_config.NumberColumn(step=1, min_value=1, format="%d")},
        key="rooms_editor",
        on_change=save_rooms_callback,
    )

c3, c4 = st.columns(2)
with c3:
    st.subheader("⛔️ Verbotene Paare (auto-save)")
    st.data_editor(
        st.session_state.forbidden_pairs,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "a": st.column_config.SelectboxColumn(options=people_ids_snapshot),
            "b": st.column_config.SelectboxColumn(options=people_ids_snapshot),
        },
        key="forbidden_editor",
        on_change=save_forbidden_callback,
    )

with c4:
    st.subheader("🧑‍🏫 Lehrkräfte pro Gang (auto-save)")
    st.data_editor(
        st.session_state.required_teachers_per_corridor,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "corridor": st.column_config.SelectboxColumn(options=corridors_snapshot),
            "teacher_id": st.column_config.SelectboxColumn(options=teacher_ids_snapshot),
        },
        key="req_editor",
        on_change=save_req_callback,
    )

# ------------------------------ PDF plan generator (two columns per card) ------------------------------
def _format_person(o: dict) -> str:
    gender = o.get("gender") or ""
    role = o.get("role") or ""
    cls = o.get("class_id")
    parts = []
    if gender:
        parts.append(gender)
    if role == "teacher":
        parts.append("teacher")
    if cls:
        parts.append(str(cls))
    meta = ", ".join(parts)
    return f"{o.get('name', o.get('id', ''))}" + (f" ({meta})" if meta else "")

def _generate_pdf_plan(allocation: dict, rooms: dict, title: str = "Zimmerplan") -> bytes:
    page_size = landscape(A4)
    W, H = page_size
    margin = 12 * mm
    cols = 5
    rows = 2
    gutter = 5 * mm
    grid_w = W - 2 * margin
    grid_h = H - 2 * margin - 18 * mm  # header space
    cell_w = (grid_w - (cols - 1) * gutter) / cols
    cell_h = (grid_h - (rows - 1) * gutter) / rows

    PAD = 3 * mm
    LINE_H = 4.5 * mm
    FS = 8.5
    FS_B = 11

    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=page_size)

    def room_sort_key(rid):
        r = rooms.get(rid, {})
        return (str(r.get("corridor", "")), str(r.get("name", rid)))
    room_ids_sorted = sorted(allocation.keys(), key=room_sort_key)

    today = date.today().strftime("%d.%m.%Y")

    def header():
        c.setFont("Helvetica-Bold", 20)
        c.drawString(margin, H - margin - 6 * mm, f"{title}")
        c.setFont("Helvetica", 11)
        txt = today
        tw = c.stringWidth(txt, "Helvetica", 11)
        c.drawString(W - margin - tw, H - margin - 6 * mm, txt)

    def wrap_text(text: str, font: str, size: float, max_w: float) -> list:
        c.setFont(font, size)
        words = text.split(" ")
        lines, cur = [], ""
        for w in words:
            cand = (cur + " " + w).strip()
            if c.stringWidth(cand, font, size) <= max_w:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def draw_card(x, y, rid):
        r = rooms.get(rid, {"name": rid, "capacity": 0, "corridor": ""})
        rname = f"{r.get('name', rid)}"
        cap = int(r.get("capacity", 0))
        occs = allocation.get(rid, [])
        used = len(occs)
        free = max(cap - used, 0)
        corridor = str(r.get("corridor", ""))

        # frame
        c.setStrokeColor(colors.black)
        c.setLineWidth(1)
        c.roundRect(x, y, cell_w, cell_h, 3 * mm, stroke=1, fill=0)

        # room header
        c.setFont("Helvetica-Bold", FS_B)
        c.drawString(x + PAD, y + cell_h - 6 * mm, rname)
        c.setFont("Helvetica", 9)
        c.drawString(x + PAD, y + cell_h - 11 * mm, f"{corridor}   ({used}/{cap}, frei: {free})")

        # content area
        content_top = y + cell_h - 18 * mm
        content_bottom = y + PAD
        content_height = content_top - content_bottom
        content_left = x + PAD
        content_width = cell_w - 2 * PAD

        # two columns
        col_gap = 3 * mm
        col_w = (content_width - col_gap) / 2
        col_x = [content_left, content_left + col_w + col_gap]

        # build single list with section headers, then wrap
        students = [o for o in occs if o.get("role") == "student"]
        teachers = [o for o in occs if o.get("role") == "teacher"]

        logical_lines = []
        if students:
            logical_lines.append(("header", "Schüler:innen:"))
            for o in students:
                logical_lines.append(("item", "• " + _format_person(o)))
        if teachers:
            logical_lines.append(("header", "Lehrkräfte:"))
            for o in teachers:
                logical_lines.append(("item", "• " + _format_person(o)))

        draw_lines = []
        for kind, text in logical_lines:
            if kind == "header":
                draw_lines.append((kind, text))
            else:
                for w in wrap_text(text, "Helvetica", FS, col_w):
                    draw_lines.append(("item", w))

        # lay out into two columns with overflow indicator
        c.setFont("Helvetica", FS)
        col_idx = 0
        cursor_y = content_top
        overflow = 0
        for kind, text in draw_lines:
            font_name = "Helvetica-Bold" if kind == "header" else "Helvetica"
            c.setFont(font_name, FS)

            if cursor_y - LINE_H < content_bottom - 0.1:
                col_idx += 1
                if col_idx >= 2:
                    overflow += 1
                    continue
                cursor_y = content_top

            c.drawString(col_x[col_idx], cursor_y, text)
            cursor_y -= LINE_H

        if overflow > 0:
            c.setFont("Helvetica-Oblique", FS)
            c.drawRightString(x + cell_w - PAD, content_bottom + 0.5 * LINE_H, f"... +{overflow} weitere")

    # paginate rooms
    header()
    col = row = 0
    start_x = margin
    start_y = H - margin - 18 * mm - cell_h

    for rid in room_ids_sorted:
        x = start_x + col * (cell_w + gutter)
        y = start_y - row * (cell_h + gutter)
        draw_card(x, y, rid)
        col += 1
        if col >= cols:
            col = 0
            row += 1
            if row >= rows:
                c.showPage()
                header()
                row = 0
                col = 0

    c.showPage()
    c.save()
    return buf.getvalue()

# ------------------------------ Helpers for solver ------------------------------
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

# ------------------------------ Result renderer (survives reruns) ------------------------------
def render_results():
    lr = st.session_state.last_result
    if not lr:
        return
    out_df = lr["out_df"]
    st.subheader("Ergebnis")
    st.dataframe(out_df, use_container_width=True)

    # PDF / CSV downloads (use cached bytes; no recompute)
    st.download_button(
        "📄 PDF-Zimmerplan herunterladen",
        data=lr["pdf_bytes"],
        file_name="zimmerplan.pdf",
        mime="application/pdf",
        type="secondary",
        key="dl_pdf",
    )

    cdl, cinfo = st.columns([1,1])
    with cdl:
        st.download_button(
            "📥 Ergebnis als CSV",
            data=lr["csv_bytes"],
            file_name="room_assignment.csv",
            mime="text/csv",
            key="dl_csv",
        )
        if show_export:
            st.download_button(
                "📥 people.json",
                st.session_state.people.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
                "people.json", "application/json", key="dl_people_json"
            )
            st.download_button(
                "📥 rooms.json",
                st.session_state.rooms.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
                "rooms.json", "application/json", key="dl_rooms_json"
            )
            st.download_button(
                "📥 forbidden_pairs.json",
                st.session_state.forbidden_pairs.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
                "forbidden_pairs.json", "application/json", key="dl_forbidden_json"
            )
            st.download_button(
                "📥 required_teachers_per_corridor.json",
                st.session_state.required_teachers_per_corridor.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8"),
                "required_teachers_per_corridor.json", "application/json", key="dl_req_json"
            )
    with cinfo:
        st.markdown("**Solver-Infos**")
        st.json(lr["meta"])

# Always render last result if available (survives reruns like download clicks)
render_results()

# ------------------------------ Run solver ------------------------------
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
            alloc = result.get("allocation", {})

            if isinstance(alloc, list) or not isinstance(alloc, dict):
                st.error("Allocation-Format unerwartet. Bitte 'solver.py' aus diesem Projekt verwenden.")
                st.stop()

            # Build result table
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

            out_df = pd.DataFrame(rows).sort_values(
                ["corridor","room_name","role","gender","name"]
            ) if rows else pd.DataFrame(
                columns=["room_id","room_name","corridor","capacity","person_id","name","role","gender","class_id"]
            )

            # PDF & CSV bytes (cached)
            pdf_bytes = _generate_pdf_plan(allocation=alloc, rooms=rooms, title="Zimmerplan")
            csv_bytes = out_df.to_csv(index=False).encode("utf-8")

            # Store into session for persistence across reruns
            st.session_state.last_result = {
                "alloc": alloc,
                "out_df": out_df,
                "pdf_bytes": pdf_bytes,
                "csv_bytes": csv_bytes,
                "rooms": rooms,
                "meta": {
                    "objective": result.get("objective"),
                    "status": result.get("status"),
                    "stats": result.get("stats"),
                },
            }

            st.success("Fertig!")
            render_results()

# ------------------------------ Footer ------------------------------
st.markdown("""
**Hinweise**
- Tabellen speichern automatisch auf **Enter/Tab** (Auto-Save per `on_change`).
- CSV-Uploads pro Tabelle überschreiben die jeweilige Tabelle und werden sofort gespeichert.
- PDF-Export erzeugt Raumkarten im A4-Querformat (5×3 Grid), **zweispaltige** Belegungsliste mit Umbruch & Overflow-Hinweis.
- Lehrkräfte können `class_id` haben; der Solver bevorzugt denselben **Flur** wie die Klasse (weich).
- Lehrkräfte werden **nie** mit Schüler:innen gemischt; bevorzugt Einzelzimmer.
""")
