# app.py
# Strength Prescriptor (MVP) — Streamlit
# - Strength Tests (1RM) per client & exercise
# - Sessions: add via single date or date range (calendar-like)
# - Sets: exercise, sets, reps, %1RM, 1RM (autofill), suggested kg (editable)
# - Edit/Delete sets
# - Per-exercise and total session tonnage
# - Export session CSV (includes TOTAL tonnage row at the end)
#
# No authentication. SQLite DB created on first run.

from __future__ import annotations
import datetime as dt
from typing import Optional, Dict
import io
import csv

import streamlit as st
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Date,
    ForeignKey, Text, DateTime, Boolean
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# -----------------------------
# DB (SQLite for MVP)
# -----------------------------
engine = create_engine("sqlite:///strength_mvp.db", echo=False, future=True)
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
    role = Column(String, default="coach")  # coach | client | admin
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
    category = Column(String)   # squat, hinge, push, pull, core, etc.
    equipment = Column(String)
    unilateral = Column(Boolean, default=False)
    description = Column(Text, default="")

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

Base.metadata.create_all(engine)

# -----------------------------
# DEMO DATA (to get started)
# -----------------------------
def init_demo_data():
    db = SessionLocal()
    if not db.query(User).first():
        coach = User(email="coach@example.com", name="Coach Demo", role="coach", hash="demo")
        client_user = User(email="client@example.com", name="Client Demo", role="client", hash="demo")
        db.add_all([coach, client_user]); db.commit()

        c = Client(user_id=client_user.id, sex="female", dob=dt.date(1990,1,1),
                   height_cm=165, weight_kg=60, owner_id=coach.id)
        db.add(c)
        # Exercises
        for ex in [
            ("Back Squat", "squat", "barbell"),
            ("Bench Press", "push", "barbell"),
            ("Deadlift", "hinge", "barbell"),
            ("Lat Pulldown", "pull", "machine"),
        ]:
            if not db.query(Exercise).filter(Exercise.name == ex[0]).first():
                db.add(Exercise(name=ex[0], category=ex[1], equipment=ex[2]))
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

# -----------------------------
# PAGES — Strength only
# -----------------------------
def page_strength_tests():
    st.title("🧪 Strength Tests (1RM)")
    db = SessionLocal()

    clients = db.query(Client).all()
    if not clients:
        st.info("Create a client first in the DB seeding.")
        db.close(); return

    client_map = {db.query(User).get(c.user_id).name: c.id for c in clients}
    client_name = st.selectbox("Client", options=list(client_map.keys()))

    exs = db.query(Exercise).all()
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

    exs = db.query(Exercise).all()
    exmap = {e.name: e.id for e in exs}

    # --- Create sessions (single date or date range) ---
    with st.expander("➕ Add sessions"):
        mode = st.radio("Add", ["Single date", "Date range (calendar)"], horizontal=True)
        if mode == "Single date":
            d = st.date_input("Date", value=plan.start_date, key="single_date")
            focus = st.text_input("Focus", value="Strength (basic)", key="single_focus")
            notes = st.text_area("Notes", value="", key="single_notes")
            if st.button("Add session", key="add_single_sess"):
                s = TrainingSession(plan_id=plan.id, date=d, focus=focus, notes=notes)
                db.add(s); db.commit()
                st.success(f"Session added for {d}")
        else:
            start = st.date_input("Start date", value=plan.start_date, key="range_start")
            end = st.date_input("End date", value=min(plan.end_date, plan.start_date+dt.timedelta(days=6)), key="range_end")
            focus = st.text_input("Focus for range", value="Strength (block)", key="range_focus")
            notes = st.text_area("Notes for range", value="", key="range_notes")
            freq = st.selectbox("Frequency", ["Daily", "Every 2 days", "Every 3 days"], key="range_freq")
            step = {"Daily": 1, "Every 2 days": 2, "Every 3 days": 3}[freq]
            if st.button("Create sessions for range", key="add_range_sess"):
                if end < start:
                    st.error("End date must be after start date.")
                else:
                    dcur = start
                    created = 0
                    while dcur <= end:
                        s = TrainingSession(plan_id=plan.id, date=dcur, focus=focus, notes=notes)
                        db.add(s)
                        created += 1
                        dcur = dcur + dt.timedelta(days=step)
                    db.commit()
                    st.success(f"Created {created} sessions from {start} to {end} ({freq})")

    # --- List sessions by date ---
    sessions = (db.query(TrainingSession)
                  .filter(TrainingSession.plan_id == plan.id)
                  .order_by(TrainingSession.date).all())

    for s in sessions:
        st.subheader(f"{s.date} — {s.focus}")
        if s.notes: st.caption(s.notes)

        # --- Add set to this session ---
        with st.expander("Add set to this session"):
            ex_name = st.selectbox(
                f"Exercise (session {s.id})", options=list(exmap.keys()),
                key=f"ex_{s.id}"
            )
            sets = st.number_input("Sets", 1, 20, 4, key=f"sets_{s.id}")
            reps = st.number_input("Reps", 1, 50, 6, key=f"reps_{s.id}")

            ex_id = exmap[ex_name]
            latest_1rm = get_latest_one_rm(db, plan.client_id, ex_id)

            col_a, col_b = st.columns(2)
            with col_a:
                pct = st.slider("Intensity (%1RM)", 30, 100, 75, step=1, key=f"pct_{s.id}")
            with col_b:
                one_rm_input = st.number_input(
                    "1RM from test (kg)", 0.0, 500.0,
                    float(latest_1rm) if latest_1rm else 100.0,
                    key=f"onerm_{s.id}"
                )

            suggested_load = round_to_05((pct / 100.0) * one_rm_input)
            load = st.number_input(
                "Load per rep (kg)", 0.0, 500.0, suggested_load,
                step=0.5, key=f"load_{s.id}"
            )
            rest = st.number_input("Rest (s)", 0, 600, 120, key=f"rest_{s.id}")
            note_in = st.text_input("Notes", value="", key=f"notes_{s.id}")

            st.caption(
                f"Suggested: {suggested_load:.1f} kg (1RM {one_rm_input:.1f}, {pct}%) — editable."
            )

            if st.button("Add set", key=f"addset_{s.id}"):
                if sets <= 0 or reps <= 0 or load <= 0:
                    st.error("Sets, reps and load must be > 0.")
                else:
                    sp = SetPrescription(
                        session_id=s.id, exercise_id=ex_id,
                        sets=int(sets), reps=int(reps),
                        intensity_pct_1rm=float(pct),
                        load_kg=float(load),
                        rest_sec=int(rest), notes=note_in,
                    )
                    db.add(sp); db.commit()
                    st.success("Set added")

        # --- Show sets with Edit/Delete + per-exercise & total tonnage ---
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
                        SessionLocal().commit()  # commit in a fresh session if needed
                        db.commit()
                        st.success("Set updated — reload the page or scroll to see recalculated tonnage.")
                if c2.button("🗑️ Delete", key=f"del_{sp.id}"):
                    db.delete(sp)
                    db.commit()
                    st.warning("Set deleted — it will disappear on refresh.")

        # Per-exercise tonnage summary
        if tonnage_by_ex:
            st.subheader("Per-exercise tonnage")
            for ex_name, ton in sorted(tonnage_by_ex.items()):
                st.write(f"• **{ex_name}**: {ton:.1f} kg")

        st.info(f"Total session tonnage: **{session_tonnage:.1f} kg**")

        # --- CSV Export (includes total tonnage row) ---
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

            # Add TOTAL row
            total_row = {k: "" for k in rows[0].keys()}
            total_row["exercise"] = "TOTAL"
            total_row["tonnage"] = round(session_tonnage, 1)
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

def page_settings():
    st.title("⚙️ Settings")
    st.write("Local SQLite DB file: `strength_mvp.db`")
    if st.button("Reset database (danger)"):
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        init_demo_data()
        st.success("Database reset.")

# -----------------------------
# LAYOUT & NAV
# -----------------------------
st.set_page_config(page_title="Strength Prescriptor (MVP)", layout="wide")
with st.sidebar:
    st.title("Strength Prescriptor (MVP)")

page = st.sidebar.radio("Navigation", ["Strength Tests", "Sessions", "Settings"])

if page == "Strength Tests":
    page_strength_tests()
elif page == "Sessions":
    page_sessions()
elif page == "Settings":
    page_settings()
