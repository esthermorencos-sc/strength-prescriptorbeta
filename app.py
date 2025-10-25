
# app.py
# Strength Prescriptor (MVP) — Streamlit
# FIX: use /tmp for SQLite & images (writable on Streamlit Cloud)
# - Visual calendar (create + drag & drop reschedule)
# - Sessions: %1RM + 1RM (autofill) -> suggested kg (editable/manual)
# - Edit/Delete sets, per-exercise & total session tonnage
# - Export CSV (TOTAL row)
# - Analytics charts
# - Exercises with image upload/display
#
# No authentication.

from __future__ import annotations
import os
import io
import csv
import datetime as dt
from typing import Optional, Dict

import pandas as pd
import altair as alt
import streamlit as st
from streamlit_calendar import calendar  # pip install streamlit-calendar

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date,
    ForeignKey, Text, DateTime, Boolean
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# -----------------------------
# Writable paths (Streamlit Cloud)
# -----------------------------
DB_PATH = "/tmp/strength_mvp.db"         # /tmp is writable in Streamlit Cloud
IMAGES_DIR = "/tmp/exercise_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# -----------------------------
# DB (SQLite for MVP)
# -----------------------------
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},  # allow usage across threads
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# -----------------------------
# MODELS
# -----------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, default="coach")
    hash = Column(String, nullable=False)   # placeholder (unused here)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

class Client(Base):
    __tablename__ = "clients"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    sex = Column(String)
    dob = Column(Date)
    height_cm = Column(Float)
    weight_kg = Column(Float)
    notes = Column(Text, default="")
    owner_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", foreign_keys=[user_id])
    owner = relationship("User", foreign_keys=[owner_id])

class Exercise(Base):
    __tablename__ = "exercises"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    category = Column(String)   # squat, hinge, push, pull, core, isolation, plyo, machine, etc.
    equipment = Column(String)
    unilateral = Column(Boolean, default=False)
    description = Column(Text, default="")
    image_path = Column(String, default="")  # local path to uploaded image (optional)

class TrainingPlan(Base):
    __tablename__ = "training_plans"
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey("clients.id"))
    name = Column(String)
    start_date = Column(Date)
    end_date = Column(Date)
    goal = Column(Text)
    client = relationship("Client")

class TrainingSession(Base):
    __tablename__ = "training_sessions"
    id = Column(Integer, primary_key=True)
    plan_id = Column(Integer, ForeignKey("training_plans.id"))
    date = Column(Date)
    focus = Column(String)
    notes = Column(Text, default="")
    plan = relationship("TrainingPlan")

class SetPrescription(Base):
    __tablename__ = "set_prescriptions"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("training_sessions.id"))
    exercise_id = Column(Integer, ForeignKey("exercises.id"))
    sets = Column(Integer)
    reps = Column(Integer)
    intensity_pct_1rm = Column(Float)  # e.g., 75 => 75%1RM
    load_kg = Column(Float)            # kg per rep
    rest_sec = Column(Integer)
    notes = Column(Text, default="")
    session = relationship("TrainingSession")
    exercise = relationship("Exercise")

class StrengthTest(Base):
    __tablename__ = "strength_tests"
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey("clients.id"))
    exercise_id = Column(Integer, ForeignKey("exercises.id"))
    date = Column(Date)
    one_rm_kg = Column(Float)   # validated/estimated 1RM
    notes = Column(Text, default="")
    client = relationship("Client")
    exercise = relationship("Exercise")

# Ensure tables exist (first run)
Base.metadata.create_all(engine)

# -----------------------------
# DEMO DATA (seed)
# -----------------------------
EXTENDED_EXERCISES = [
    # Barbell compounds
    ("Back Squat", "squat", "barbell", False),
    ("Front Squat", "squat", "barbell", False),
    ("Bench Press", "push", "barbell", False),
    ("Incline Bench Press", "push", "barbell", False),
    ("Overhead Press", "push", "barbell", False),
    ("Deadlift", "hinge", "barbell", False),
    ("Romanian Deadlift", "hinge", "barbell", False),
    ("Barbell Row", "pull", "barbell", False),
    # Dumbbell
    ("Dumbbell Bench Press", "push", "dumbbell", False),
    ("Dumbbell Row", "pull", "dumbbell", True),
    ("Goblet Squat", "squat", "dumbbell", False),
    ("Dumbbell Shoulder Press", "push", "dumbbell", False),
    ("Lateral Raise", "isolation", "dumbbell", False),
    ("Biceps Curl", "isolation", "dumbbell", False),
    ("Triceps Extension", "isolation", "dumbbell", False),
    ("Bulgarian Split Squat", "squat", "dumbbell", True),
    # Kettlebell
    ("Kettlebell Swing", "hinge", "kettlebell", False),
    ("Turkish Get-Up", "core", "kettlebell", True),
    # Machine / Cable
    ("Lat Pulldown", "pull", "machine", False),
    ("Seated Row", "pull", "machine", False),
    ("Leg Press", "squat", "machine", False),
    ("Leg Extension", "isolation", "machine", False),
    ("Leg Curl", "isolation", "machine", False),
    ("Pec Deck", "isolation", "machine", False),
    ("Cable Row", "pull", "cable", False),
    ("Cable Fly", "isolation", "cable", False),
    ("Cable Lateral Raise", "isolation", "cable", False),
    # Bodyweight
    ("Pull-Up", "pull", "bodyweight", False),
    ("Chin-Up", "pull", "bodyweight", False),
    ("Push-Up", "push", "bodyweight", False),
    ("Dip", "push", "bodyweight", False),
    ("Plank", "core", "bodyweight", False),
    ("Hip Thrust", "hinge", "bodyweight", False),
]

def init_demo_data():
    db = SessionLocal()
    if not db.query(User).first():
        coach = User(email="coach@example.com", name="Coach Demo", role="coach", hash="demo")
        client_user = User(email="client@example.com", name="Client Demo", role="client", hash="demo")
        db.add_all([coach, client_user]); db.commit()

        c = Client(user_id=client_user.id, sex="female", dob=dt.date(1990,1,1),
                   height_cm=165, weight_kg=60, owner_id=coach.id)
        db.add(c)
        # Extended exercises (deduplicated)
        for name, cat, equip, uni in EXTENDED_EXERCISES:
            if not db.query(Exercise).filter(Exercise.name == name).first():
                db.add(Exercise(name=name, category=cat, equipment=equip, unilateral=uni))
        db.commit()
        # Plan
        if not db.query(TrainingPlan).first():
            plan = TrainingPlan(client_id=c.id, name="Preseason 4 weeks",
                                start_date=dt.date.today(),
                                end_date=dt.date.today()+dt.timedelta(days=27),
                                goal="General strength development")
            db.add(plan); db.commit()
    db.close()

init_demo_data()

# -----------------------------
# HELPERS
# -----------------------------
def get_latest_one_rm(db, client_id: int, exercise_id: int) -> Optional[float]:
    test = (db.query(StrengthTest)
              .filter(StrengthTest.client_id == client_id,
                      StrengthTest.exercise_id == exercise_id)
              .order_by(StrengthTest.date.desc())
              .first())
    return test.one_rm_kg if test else None

def compute_tonnage(sets: int, reps: int, load_kg: float) -> float:
    s = max(0, int(sets or 0))
    r = max(0, int(reps or 0))
    w = max(0.0, float(load_kg or 0.0))
    return float(s * r * w)

def round_to_05(x: float) -> float:
    return round(x * 2) / 2.0

def iso_year_week(d: dt.date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"

# -----------------------------
# PAGES
# -----------------------------
def page_exercises():
    st.title("🏋️ Exercise Library (with images)")
    db = SessionLocal()

    # List
    exs = db.query(Exercise).order_by(Exercise.name).all()
    for e in exs:
        cols = st.columns([1, 3])
        with cols[0]:
            if e.image_path and os.path.exists(e.image_path):
                st.image(e.image_path, use_column_width=True)
            else:
                st.caption("No image")
        with cols[1]:
            st.markdown(f"**{e.name}** — {e.category} · {e.equipment}{' · unilateral' if e.unilateral else ''}")
            if e.description:
                st.caption(e.description)

    st.divider()
    with st.expander("➕ Add / Update exercise"):
        name = st.text_input("Name")
        category = st.selectbox("Category", sorted({x[1] for x in EXTENDED_EXERCISES} | {"core","isolation","plyo","machine","cable","bodyweight"}))
        equipment = st.text_input("Equipment", value="barbell")
        unilateral = st.checkbox("Unilateral", value=False)
        description = st.text_area("Description", value="")
        image_file = st.file_uploader("Upload image (png/jpg)", type=["png","jpg","jpeg"])
        if st.button("Save exercise"):
            if not name.strip():
                st.error("Name is required.")
            else:
                existing = db.query(Exercise).filter(Exercise.name == name).first()
                if existing:
                    # update
                    existing.category = category
                    existing.equipment = equipment
                    existing.unilateral = unilateral
                    existing.description = description
                    db.commit()
                    ex_obj = existing
                else:
                    ex_obj = Exercise(name=name, category=category, equipment=equipment,
                                      unilateral=unilateral, description=description)
                    db.add(ex_obj); db.commit()
                # handle image
                if image_file is not None:
                    safe_name = f"{ex_obj.id}_{image_file.name}".replace(" ", "_")
                    save_path = os.path.join(IMAGES_DIR, safe_name)
                    with open(save_path, "wb") as f:
                        f.write(image_file.read())
                    ex_obj.image_path = save_path
                    db.commit()
                st.success("Exercise saved. Reload to see updates.")
    db.close()

def page_strength_tests():
    st.title("🧪 Strength Tests (1RM)")
    db = SessionLocal()

    clients = db.query(Client).all()
    if not clients:
        st.info("Seed created a demo client automatically.")
        db.close(); return

    client_map = {db.query(User).get(c.user_id).name: c.id for c in clients}
    client_name = st.selectbox("Client", options=list(client_map.keys()))

    exs = db.query(Exercise).order_by(Exercise.name).all()
    if not exs:
        st.info("Add exercises first.")
        db.close(); return
    exmap = {e.name: e.id for e in exs}
    ex_name = st.selectbox("Exercise", options=list(exmap.keys()))

    date = st.date_input("Test date", value=dt.date.today())
    onerm = st.number_input("1RM (kg)", 0.0, 500.0, 100.0)
    notes = st.text_area("Notes", "")

    if st.button("Save 1RM test"):
        t = StrengthTest(client_id=client_map[client_name],
                         exercise_id=exmap[ex_name],
                         date=date, one_rm_kg=onerm, notes=notes)
        db.add(t); db.commit()
        st.success("Test saved")

    st.subheader("Recent history")
    tests = (
        db.query(StrengthTest)
          .filter(StrengthTest.client_id == client_map[client_name])
          .order_by(StrengthTest.date.desc())
          .limit(20).all()
    )
    for t in tests:
        ex = db.query(Exercise).get(t.exercise_id)
        st.write(f"{t.date} — {ex.name}: **{t.one_rm_kg:.1f} kg**")
    db.close()

def page_calendar():
    st.title("📆 Calendar — Add & Reschedule sessions (drag & drop)")
    db = SessionLocal()

    # Select plan
    plans = db.query(TrainingPlan).all()
    if not plans:
        st.info("A demo plan was created automatically in seeding.")
        db.close(); return

    plan_map = {f"{p.name} ({p.start_date}→{p.end_date})": p.id for p in plans}
    psel = st.selectbox("Plan", options=list(plan_map.keys()))
    plan = db.query(TrainingPlan).get(plan_map[psel])

    # Build events from existing sessions
    sessions = (db.query(TrainingSession)
                  .filter(TrainingSession.plan_id == plan.id)
                  .order_by(TrainingSession.date).all())
    events = []
    for s in sessions:
        events.append({
            "id": str(s.id),
            "title": s.focus or "Session",
            "start": s.date.isoformat(),
            "end": (s.date + dt.timedelta(days=1)).isoformat(),
        })

    st.caption("Click a day to add a session. Drag & drop a session to reschedule. Configure sets in the Sessions page.")
    cal_options = {
        "editable": True,          # enable drag & drop
        "selectable": True,        # enable selecting/creating
        "initialView": "dayGridMonth",
        "height": 700,
    }
    cal = calendar(events=events, options=cal_options)

    # Handle interactions
    if cal and isinstance(cal, dict):
        # date click -> create new session
        if "date" in cal and cal.get("action") == "dateClick":
            try:
                date = dt.date.fromisoformat(cal["date"][:10])
                st.success(f"Selected {date}. Create session below:")
                focus = st.text_input("Focus", value="Strength (calendar)")
                notes = st.text_area("Notes", value="")
                if st.button("Create session on selected date"):
                    new_sess = TrainingSession(plan_id=plan.id, date=date, focus=focus, notes=notes)
                    db.add(new_sess); db.commit()
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Couldn't parse date: {e}")

        # eventDrop (drag & drop) — normalize expected payloads
        ev = None
        if "eventDrop" in cal:
            ev = cal.get("eventDrop", {}).get("event")
        elif cal.get("action") == "eventDrop":
            ev = cal.get("event")
        # Update session date if event found
        if ev and isinstance(ev, dict):
            sid = ev.get("id")
            start = ev.get("start")
            try:
                if sid and start:
                    new_date = dt.date.fromisoformat(start[:10])
                    sess = db.query(TrainingSession).get(int(sid))
                    if sess and sess.plan_id == plan.id:
                        sess.date = new_date
                        db.commit()
                        st.success(f"Session #{sid} moved to {new_date}.")
                        st.experimental_rerun()
            except Exception as e:
                st.error(f"Couldn't reschedule: {e}")
    db.close()

def page_sessions():
    st.title("🗓️ Sessions & Sets (Tonnage)")
    db = SessionLocal()

    plans = db.query(TrainingPlan).all()
    if not plans:
        st.info("Create a training plan first (seeded demo created one).")
        db.close(); return

    plan_map = {f"{p.name} ({p.start_date}→{p.end_date})": p.id for p in plans}
    psel = st.selectbox("Plan", options=list(plan_map.keys()))
    plan = db.query(TrainingPlan).get(plan_map[psel])
    st.caption(plan.goal)

    exs = db.query(Exercise).order_by(Exercise.name).all()
    exmap = {e.name: e.id for e in exs}

    # Quick add session
    with st.expander("➕ Add session (quick)"):
        d = st.date_input("Date", value=plan.start_date)
        focus = st.text_input("Focus", value="Strength (basic)")
        notes = st.text_area("Notes", value="")
        if st.button("Add session"):
            s = TrainingSession(plan_id=plan.id, date=d, focus=focus, notes=notes)
            db.add(s); db.commit()
            st.success(f"Session added for {d}")

    # List sessions
    sessions = (db.query(TrainingSession)
                  .filter(TrainingSession.plan_id == plan.id)
                  .order_by(TrainingSession.date).all())

    for s in sessions:
        st.subheader(f"{s.date} — {s.focus}")
        if s.notes: st.caption(s.notes)

        # Add set to this session
        with st.expander("Add set to this session"):
            ex_name = st.selectbox(
                f"Exercise (session {s.id})", options=list(exmap.keys()),
                key=f"ex_{s.id}"
            )
            sets = st.number_input("Sets", 1, 20, 4, key=f"sets_{s.id}")
            reps = st.number_input("Reps", 1, 50, 6, key=f"reps_{s.id}")

            ex_id = exmap[ex_name]
            latest_1rm = get_latest_one_rm(db, plan.client_id, ex_id)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                pct = st.slider("Intensity (%1RM)", 30, 100, 75, step=1, key=f"pct_{s.id}")
            with col_b:
                one_rm_input = st.number_input(
                    "1RM from test (kg)", 0.0, 500.0,
                    float(latest_1rm) if latest_1rm else 100.0,
                    key=f"onerm_{s.id}"
                )
            with col_c:
                mode = st.radio("Load mode", ["Calculate from %1RM", "Enter kg manually"], key=f"mode_{s.id}", horizontal=False)

            suggested_load = round_to_05((pct / 100.0) * one_rm_input)
            if mode == "Calculate from %1RM":
                load = st.number_input(
                    "Load per rep (kg)", 0.0, 500.0, suggested_load,
                    step=0.5, key=f"load_{s.id}"
                )
            else:
                load = st.number_input(
                    "Load per rep (kg)", 0.0, 500.0, 60.0,
                    step=0.5, key=f"load_{s.id}"
                )
            rest = st.number_input("Rest (s)", 0, 600, 120, key=f"rest_{s.id}")
            note_in = st.text_input("Notes", value="", key=f"notes_{s.id}")

            st.caption(
                f"Suggested by %1RM: {suggested_load:.1f} kg (1RM {one_rm_input:.1f}, {pct}%). You can override manually."
            )

            if st.button("Add set", key=f"addset_{s.id}"):
                if sets <= 0 or reps <= 0 or load <= 0:
                    st.error("Sets, reps and load must be > 0.")
                else:
                    sp = SetPrescription(
                        session_id=s.id, exercise_id=ex_id,
                        sets=int(sets), reps=int(reps),
                        intensity_pct_1rm=float(pct),
                        load_kg=round_to_05(float(load)),
                        rest_sec=int(rest), notes=note_in,
                    )
                    db.add(sp); db.commit()
                    st.success("Set added")

        # Show sets with Edit/Delete + tonnage
        sets_ = db.query(SetPrescription).filter(SetPrescription.session_id == s.id).all()
        session_tonnage = 0.0
        tonnage_by_ex: Dict[str, float] = {}

        for sp in sets_:
            e = db.query(Exercise).get(sp.exercise_id)
            tonnage = compute_tonnage(sp.sets, sp.reps, sp.load_kg)
            session_tonnage += tonnage
            tonnage_by_ex[e.name] = tonnage_by_ex.get(e.name, 0.0) + tonnage

            st.write(
                f"• {e.name}: {sp.sets}×{sp.reps} @ {sp.load_kg:.1f} kg "
                f"(~{sp.intensity_pct_1rm or 0:.0f}%1RM) — rest {sp.rest_sec}s "
                f"— **Tonnage: {tonnage:.1f} kg**"
            )
            if sp.notes:
                st.caption(sp.notes)

            with st.expander(f"Edit/Delete set #{sp.id}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    new_sets = st.number_input("Sets", 1, 50, sp.sets, key=f"edit_sets_{sp.id}")
                    new_reps = st.number_input("Reps", 1, 100, sp.reps, key=f"edit_reps_{sp.id}")
                with col2:
                    new_pct = st.slider("%1RM", 30, 100, int(sp.intensity_pct_1rm or 75), key=f"edit_pct_{sp.id}")
                    new_load = st.number_input("Load (kg)", 0.0, 500.0, float(sp.load_kg or 0.0), step=0.5, key=f"edit_load_{sp.id}")
                with col3:
                    new_rest = st.number_input("Rest (s)", 0, 600, int(sp.rest_sec or 0), key=f"edit_rest_{sp.id}")
                new_notes = st.text_input("Notes", value=sp.notes or "", key=f"edit_notes_{sp.id}")

                c1, c2 = st.columns(2)
                if c1.button("💾 Save changes", key=f"save_{sp.id}"):
                    if new_sets <= 0 or new_reps <= 0 or new_load <= 0:
                        st.error("Sets, reps and load must be > 0.")
                    else:
                        sp.sets = int(new_sets)
                        sp.reps = int(new_reps)
                        sp.intensity_pct_1rm = float(new_pct)
                        sp.load_kg = round_to_05(float(new_load))
                        sp.rest_sec = int(new_rest)
                        sp.notes = new_notes
                        db.commit()
                        st.success("Set updated.")
                if c2.button("🗑️ Delete", key=f"del_{sp.id}"):
                    db.delete(sp); db.commit()
                    st.warning("Set deleted — refresh to hide it.")

        # Per-exercise tonnage
        if tonnage_by_ex:
            st.subheader("Per-exercise tonnage")
            for ex_name, ton in sorted(tonnage_by_ex.items()):
                st.write(f"• **{ex_name}**: {ton:.1f} kg")

        st.info(f"Total session tonnage: **{session_tonnage:.1f} kg**")

        # CSV Export with TOTAL row
        if sets_:
            rows = []
            for sp in sets_:
                e = db.query(Exercise).get(sp.exercise_id)
                rows.append({
                    "session_id": s.id,
                    "date": s.date.isoformat(),
                    "focus": s.focus,
                    "exercise": e.name,
                    "sets": sp.sets,
                    "reps": sp.reps,
                    "pct_1rm": sp.intensity_pct_1rm,
                    "load_kg": sp.load_kg,
                    "rest_sec": sp.rest_sec,
                    "notes": sp.notes or "",
                    "tonnage": compute_tonnage(sp.sets, sp.reps, sp.load_kg),
                })
            total_row = {k: "" for k in rows[0].keys()}
            total_row["exercise"] = "TOTAL"
            total_row["tonnage"] = round(sum(r["tonnage"] for r in rows), 1)
            rows.append(total_row)

            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader(); writer.writerows(rows)
            st.download_button(
                label="⬇️ Export session to CSV (with TOTAL)",
                data=buf.getvalue(),
                file_name=f"session_{s.id}.csv",
                mime="text/csv",
            )
    db.close()

def page_analytics():
    st.title("📊 Analytics — Tonnage per session & weekly cumulative")
    db = SessionLocal()

    plans = db.query(TrainingPlan).all()
    if not plans:
        st.info("No plans available.")
        db.close(); return
    plan_map = {f"{p.name} ({p.start_date}→{p.end_date})": p.id for p in plans}
    psel = st.selectbox("Plan", options=list(plan_map.keys()))
    plan = db.query(TrainingPlan).get(plan_map[psel])

    # Build dataframe of session tonnage
    sess = (db.query(TrainingSession)
              .filter(TrainingSession.plan_id == plan.id)
              .order_by(TrainingSession.date).all())
    rows = []
    for s in sess:
        sets_ = db.query(SetPrescription).filter(SetPrescription.session_id == s.id).all()
        tot = sum(compute_tonnage(sp.sets, sp.reps, sp.load_kg) for sp in sets_)
        rows.append({"date": s.date, "session_id": s.id, "tonnage": tot})
    db.close()

    if not rows:
        st.info("No sessions yet.")
        return

    df = pd.DataFrame(rows).sort_values("date")
    df["week"] = df["date"].apply(iso_year_week)
    weekly = df.groupby("week", as_index=False)["tonnage"].sum()
    weekly["cum_tonnage"] = weekly["tonnage"].cumsum()

    st.subheader("Tonnage per session")
    chart1 = alt.Chart(df).mark_bar().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("tonnage:Q", title="Tonnage (kg)"),
        tooltip=["date:T","tonnage:Q","session_id:Q"]
    ).properties(height=300)
    st.altair_chart(chart1, use_container_width=True)

    st.subheader("Weekly cumulative tonnage")
    chart2 = alt.Chart(weekly).mark_line(point=True).encode(
        x=alt.X("week:N", title="ISO week"),
        y=alt.Y("cum_tonnage:Q", title="Cumulative Tonnage (kg)"),
        tooltip=["week:N","tonnage:Q","cum_tonnage:Q"]
    ).properties(height=300)
    st.altair_chart(chart2, use_container_width=True)

def page_settings():
    st.title("⚙️ Settings")
    st.write("SQLite file path: `/tmp/strength_mvp.db` (ephemeral)")
    if st.button("Reset database (danger)"):
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        init_demo_data()
        st.success("Database reset.")

# -----------------------------
# APP LAYOUT & NAV
# -----------------------------
st.set_page_config(page_title="Strength Prescriptor (MVP)", layout="wide")
with st.sidebar:
    st.title("Strength Prescriptor (MVP)")

page = st.sidebar.radio(
    "Navigation",
    ["Calendar", "Sessions", "Strength Tests", "Exercises", "Analytics", "Settings"]
)

if page == "Calendar":
    page_calendar()
elif page == "Sessions":
    page_sessions()
elif page == "Strength Tests":
    page_strength_tests()
elif page == "Exercises":
    page_exercises()
elif page == "Analytics":
    page_analytics()
elif page == "Settings":
    page_settings()
