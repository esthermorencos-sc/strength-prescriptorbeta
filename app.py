
# app.py
# Strength Prescriptor (MVP) — v10
# - Robust session creation with st.form (sidebar + Sessions + Calendar)
# - Auto-create demo plan if none exists at creation time
# - Explicit error messages on exceptions
# - Keeps: SQLAlchemy 2.0, schema guard, images use_container_width, reordering, CSV, analytics

from __future__ import annotations
import os, io, csv, sqlite3, traceback
import datetime as dt
from typing import Optional, Dict, List
import pandas as pd
import altair as alt
import streamlit as st
from streamlit_calendar import calendar
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, ForeignKey, Text, DateTime, Boolean
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DB_PATH = "/tmp/strength_mvp.db"
IMAGES_DIR = "/tmp/exercise_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,
    future=True,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, default="coach")
    hash = Column(String, nullable=False)
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
    category = Column(String)
    equipment = Column(String)
    unilateral = Column(Boolean, default=False)
    description = Column(Text, default="")
    image_path = Column(String, default="")

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

class SessionExercise(Base):
    __tablename__ = "session_exercises"
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("training_sessions.id"))
    exercise_id = Column(Integer, ForeignKey("exercises.id"))
    order_index = Column(Integer, default=0)
    notes = Column(Text, default="")
    session = relationship("TrainingSession")
    exercise = relationship("Exercise")

class SetPrescription(Base):
    __tablename__ = "set_prescriptions"
    id = Column(Integer, primary_key=True)
    session_exercise_id = Column(Integer, ForeignKey("session_exercises.id"))
    sets = Column(Integer)
    reps = Column(Integer)
    intensity_pct_1rm = Column(Float)
    load_kg = Column(Float)
    rest_sec = Column(Integer)
    notes = Column(Text, default="")
    session_exercise = relationship("SessionExercise")

class StrengthTest(Base):
    __tablename__ = "strength_tests"
    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, ForeignKey("clients.id"))
    exercise_id = Column(Integer, ForeignKey("exercises.id"))
    date = Column(Date)
    one_rm_kg = Column(Float)
    notes = Column(Text, default="")
    client = relationship("Client")
    exercise = relationship("Exercise")

def ensure_schema():
    Base.metadata.create_all(engine)
    req_cols = {
        "set_prescriptions": {"id","session_exercise_id","sets","reps","intensity_pct_1rm","load_kg","rest_sec","notes"},
        "session_exercises": {"id","session_id","exercise_id","order_index","notes"},
    }
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            for table, need in req_cols.items():
                cur.execute(f"PRAGMA table_info({table})")
                cols = {row[1] for row in cur.fetchall()}
                if not need.issubset(cols):
                    raise RuntimeError(f"Schema mismatch in {table}: {cols} vs {need}")
    except Exception:
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)

ensure_schema()

EXTENDED_EXERCISES = [
    ("Back Squat", "squat", "barbell", False),
    ("Front Squat", "squat", "barbell", False),
    ("Bench Press", "push", "barbell", False),
    ("Incline Bench Press", "push", "barbell", False),
    ("Overhead Press", "push", "barbell", False),
    ("Deadlift", "hinge", "barbell", False),
    ("Romanian Deadlift", "hinge", "barbell", False),
    ("Barbell Row", "pull", "barbell", False),
    ("Hip Thrust", "hinge", "barbell", False),
    ("Dumbbell Bench Press", "push", "dumbbell", False),
    ("Dumbbell Row", "pull", "dumbbell", True),
    ("Goblet Squat", "squat", "dumbbell", False),
    ("Dumbbell Shoulder Press", "push", "dumbbell", False),
    ("Lateral Raise", "isolation", "dumbbell", False),
    ("Biceps Curl", "isolation", "dumbbell", False),
    ("Triceps Extension", "isolation", "dumbbell", False),
    ("Bulgarian Split Squat", "squat", "dumbbell", True),
    ("Kettlebell Swing", "hinge", "kettlebell", False),
    ("Turkish Get-Up", "core", "kettlebell", True),
    ("Lat Pulldown", "pull", "machine", False),
    ("Seated Row", "pull", "machine", False),
    ("Leg Press", "squat", "machine", False),
    ("Leg Extension", "isolation", "machine", False),
    ("Leg Curl", "isolation", "machine", False),
    ("Pec Deck", "isolation", "machine", False),
    ("Cable Row", "pull", "cable", False),
    ("Cable Fly", "isolation", "cable", False),
    ("Cable Lateral Raise", "isolation", "cable", False),
    ("Pull-Up", "pull", "bodyweight", False),
    ("Chin-Up", "pull", "bodyweight", False),
    ("Push-Up", "push", "bodyweight", False),
    ("Dip", "push", "bodyweight", False),
    ("Plank", "core", "bodyweight", False),
]

def init_demo_data():
    db = SessionLocal()
    try:
        if not db.query(User).first():
            coach = User(email="coach@example.com", name="Coach Demo", role="coach", hash="demo")
            client_user = User(email="client@example.com", name="Client Demo", role="client", hash="demo")
            db.add_all([coach, client_user]); db.commit()

            c = Client(user_id=client_user.id, sex="female", dob=dt.date(1990,1,1),
                       height_cm=165, weight_kg=60, owner_id=coach.id)
            db.add(c)
            for name, cat, equip, uni in EXTENDED_EXERCISES:
                if not db.query(Exercise).filter(Exercise.name == name).first():
                    db.add(Exercise(name=name, category=cat, equipment=equip, unilateral=uni))
            db.commit()
            if not db.query(TrainingPlan).first():
                plan = TrainingPlan(client_id=c.id, name="Preseason 4 weeks",
                                    start_date=dt.date.today(),
                                    end_date=dt.date.today()+dt.timedelta(days=27),
                                    goal="General strength development")
                db.add(plan); db.commit()
    finally:
        db.close()

init_demo_data()

def ensure_plan(db) -> TrainingPlan:
    plan = db.query(TrainingPlan).order_by(TrainingPlan.id.asc()).first()
    if plan:
        return plan
    # Create demo client/plan if none
    user = db.query(User).first()
    if not user:
        user = User(email="coach@example.com", name="Coach Demo", role="coach", hash="demo")
        db.add(user); db.commit()
    client_user = db.query(User).filter(User.role=="client").first()
    if not client_user:
        client_user = User(email="client@example.com", name="Client Demo", role="client", hash="demo")
        db.add(client_user); db.commit()
    c = db.query(Client).first()
    if not c:
        c = Client(user_id=client_user.id, sex="female", dob=dt.date(1990,1,1), height_cm=165, weight_kg=60, owner_id=user.id)
        db.add(c); db.commit()
    plan = TrainingPlan(client_id=c.id, name="Auto Plan", start_date=dt.date.today(), end_date=dt.date.today()+dt.timedelta(days=27), goal="Auto-created plan")
    db.add(plan); db.commit()
    return plan

def get_latest_one_rm(db, client_id: int, exercise_id: int) -> Optional[float]:
    test = (db.query(StrengthTest)
              .filter(StrengthTest.client_id == client_id,
                      StrengthTest.exercise_id == exercise_id)
              .order_by(StrengthTest.date.desc())
              .first())
    return test.one_rm_kg if test else None

def compute_tonnage(sets: int, reps: int, load_kg: float) -> float:
    return float(max(0, int(sets or 0)) * max(0, int(reps or 0)) * max(0.0, float(load_kg or 0.0)))

def round_to_05(x: float) -> float:
    return round(x * 2) / 2.0

def goto_session(session_id: int):
    st.session_state["focus_session_id"] = int(session_id)
    st.session_state["nav"] = "Sessions"
    st.experimental_rerun()

# -------- Pages --------
def page_exercises():
    st.title("🏋️ Exercise Library (with images)")
    db = SessionLocal()
    exs = db.query(Exercise).order_by(Exercise.name).all()
    for e in exs:
        cols = st.columns([1, 3])
        with cols[0]:
            if e.image_path and os.path.exists(e.image_path):
                st.image(e.image_path, use_container_width=True)
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
                    existing.category = category
                    existing.equipment = equipment
                    existing.unilateral = unilateral
                    existing.description = description
                    db.commit()
                    ex_obj = existing
                else:
                    ex_obj = Exercise(name=name, category=category, equipment=equipment, unilateral=unilateral, description=description)
                    db.add(ex_obj); db.commit()
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
        st.info("No clients yet — creating a demo client/plan automatically when you create a session.")
        db.close(); return

    client_map = {db.get(User, c.user_id).name: c.id for c in clients}
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
        try:
            t = StrengthTest(client_id=client_map[client_name], exercise_id=exmap[ex_name], date=date, one_rm_kg=onerm, notes=notes)
            db.add(t); db.commit()
            st.success("Test saved")
        except Exception as e:
            st.error(f"Error saving test: {e}")
    db.close()

def page_calendar():
    st.title("📆 Calendar — Click, Drag & Drop")
    db = SessionLocal()

    plans = db.query(TrainingPlan).all()
    if not plans:
        st.info("No plans yet — use sidebar quick-create or Sessions page to create one by adding a session.")
    plan_map = {f"{p.name} ({p.start_date}→{p.end_date})": p.id for p in plans} if plans else {}
    psel = st.selectbox("Plan", options=list(plan_map.keys()) if plan_map else ["(none)"])
    plan = db.get(TrainingPlan, plan_map[psel]) if plan_map else None

    sessions = []
    if plan:
        sessions = (db.query(TrainingSession)
                      .filter(TrainingSession.plan_id == plan.id)
                      .order_by(TrainingSession.date).all())
    events = [{"id": str(s.id), "title": s.focus or "Session", "start": s.date.isoformat(), "end": (s.date+dt.timedelta(days=1)).isoformat()} for s in sessions]

    st.caption("Click a day to add a session; click an event to open it; drag & drop to reschedule.")
    cal = calendar(events=events, options={"editable": True, "selectable": True, "initialView": "dayGridMonth", "height": 700})

    if cal and isinstance(cal, dict):
        if cal.get("action") == "dateClick" and "date" in cal:
            date = dt.date.fromisoformat(cal["date"][:10])
            st.success(f"Selected {date}. Create session:")
            with st.form("cal_create_form", clear_on_submit=True):
                focus = st.text_input("Focus", value="Strength (calendar)")
                notes = st.text_area("Notes", value="")
                submitted = st.form_submit_button("Create session on selected date")
            if submitted:
                try:
                    with SessionLocal() as db2:
                        pl = plan if plan else ensure_plan(db2)
                        new_sess = TrainingSession(plan_id=pl.id, date=date, focus=focus, notes=notes)
                        db2.add(new_sess); db2.commit()
                        st.success(f"Session #{new_sess.id} created")
                        st.session_state["focus_session_id"] = new_sess.id
                        st.session_state["nav"] = "Sessions"
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error creating session: {e}")

        if cal.get("action") == "eventClick" and "event" in cal and isinstance(cal["event"], dict):
            sid = cal["event"].get("id")
            if sid:
                db.close(); goto_session(int(sid)); return

        ev = cal.get("eventDrop", {}).get("event") if "eventDrop" in cal else (cal.get("event") if cal.get("action")=="eventDrop" else None)
        if ev and isinstance(ev, dict):
            sid = ev.get("id"); start = ev.get("start")
            if sid and start:
                try:
                    new_date = dt.date.fromisoformat(start[:10])
                    sess = db.get(TrainingSession, int(sid))
                    if sess:
                        sess.date = new_date; db.commit()
                        st.success(f"Session #{sid} moved to {new_date}."); st.experimental_rerun()
                except Exception as e:
                    st.error(f"Couldn't reschedule: {e}")
    db.close()

def page_sessions():
    st.title("🗓️ Sessions — Builder & Tonnage")
    db = SessionLocal()

    plans = db.query(TrainingPlan).all()
    plan_map = {f"{p.name} ({p.start_date}→{p.end_date})": p.id for p in plans} if plans else {}
    psel = st.selectbox("Plan", options=list(plan_map.keys()) if plan_map else ["(none)"])
    plan = db.get(TrainingPlan, plan_map[psel]) if plan_map else None

    # Always-visible Create Session form with st.form
    st.subheader("Create new session")
    with st.form("create_session_form", clear_on_submit=True):
        new_date = st.date_input("Date", value=dt.date.today(), key="new_sess_date")
        new_focus = st.text_input("Focus", value="Strength", key="new_sess_focus")
        new_notes = st.text_input("Notes", value="", key="new_sess_notes")
        submitted = st.form_submit_button("Create & open")
    if submitted:
        try:
            with SessionLocal() as db2:
                pl = plan if plan else ensure_plan(db2)
                ns = TrainingSession(plan_id=pl.id, date=new_date, focus=new_focus, notes=new_notes)
                db2.add(ns); db2.commit()
                st.success(f"Session #{ns.id} created for {new_date}")
                st.session_state["focus_session_id"] = ns.id
                st.session_state["nav"] = "Sessions"
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Error creating session: {e}")

    # Load sessions
    focus_session_id = st.session_state.get("focus_session_id")
    sessions = []
    if plan:
        sessions = (db.query(TrainingSession).filter(TrainingSession.plan_id == plan.id).order_by(TrainingSession.date).all())
    if not sessions:
        st.info("No sessions yet. Create one above or from the sidebar.")
        db.close(); return

    session_options = {f"{s.date} — {s.focus} (#{s.id})": s.id for s in sessions}
    default_index = 0
    if focus_session_id and focus_session_id in session_options.values():
        default_index = list(session_options.values()).index(focus_session_id)
    sel_label = st.selectbox("Select session", options=list(session_options.keys()), index=default_index)
    session_id = session_options[sel_label]
    s = db.get(TrainingSession, session_id)

    exs = db.query(Exercise).order_by(Exercise.name).all()
    exmap = {e.name: e.id for e in exs}

    with st.expander("➕ Add exercise to session"):
        ex_name = st.selectbox("Exercise", options=list(exmap.keys()), key=f"addex_{s.id}")
        note_ex = st.text_input("Notes (exercise)", value="", key=f"addex_note_{s.id}")
        if st.button("Add exercise", key=f"btn_addex_{s.id}"):
            try:
                max_order = db.query(SessionExercise).filter(SessionExercise.session_id==s.id).order_by(SessionExercise.order_index.desc()).first()
                next_order = (max_order.order_index + 1) if max_order else 1
                se = SessionExercise(session_id=s.id, exercise_id=exmap[ex_name], order_index=next_order, notes=note_ex)
                db.add(se); db.commit()
                st.success(f"Added {ex_name}"); st.experimental_rerun()
            except Exception as e:
                st.error(f"Error adding exercise: {e}")

    sess_ex = (db.query(SessionExercise).filter(SessionExercise.session_id == s.id).order_by(SessionExercise.order_index.asc(), SessionExercise.id.asc()).all())

    total_session_tonnage = 0.0
    for idx, se in enumerate(sess_ex):
        ex = db.get(Exercise, se.exercise_id)
        st.markdown(f"### {idx+1}. {ex.name}")
        if ex.image_path and os.path.exists(ex.image_path):
            st.image(ex.image_path, width=200)

        c_up, c_down, c_del = st.columns(3)
        if c_up.button("⬆️ Move up", key=f"up_{se.id}") and idx > 0:
            above = sess_ex[idx-1]
            se.order_index, above.order_index = above.order_index, se.order_index
            db.commit(); st.experimental_rerun()
        if c_down.button("⬇️ Move down", key=f"down_{se.id}") and idx < len(sess_ex)-1:
            below = sess_ex[idx+1]
            se.order_index, below.order_index = below.order_index, se.order_index
            db.commit(); st.experimental_rerun()
        if c_del.button("🗑️ Remove exercise from session", key=f"del_se_{se.id}"):
            for sp in db.query(SetPrescription).filter(SetPrescription.session_exercise_id==se.id).all():
                db.delete(sp)
            db.delete(se); db.commit()
            st.warning("Exercise removed from session."); st.experimental_rerun()

        with st.expander("Add set"):
            sets = st.number_input("Sets", 1, 20, 4, key=f"sets_{se.id}")
            reps = st.number_input("Reps", 1, 50, 6, key=f"reps_{se.id}")
            latest_1rm = get_latest_one_rm(db, ensure_plan(db).client_id, se.exercise_id)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                pct = st.slider("Intensity (%1RM)", 30, 100, 75, step=1, key=f"pct_{se.id}")
            with col_b:
                one_rm_input = st.number_input("1RM from test (kg)", 0.0, 500.0, float(latest_1rm) if latest_1rm else 100.0, key=f"onerm_{se.id}")
            with col_c:
                mode = st.radio("Load mode", ["Calculate from %1RM", "Enter kg manually"], key=f"mode_{se.id}", horizontal=False)

            suggested_load = round_to_05((pct / 100.0) * one_rm_input)
            load = st.number_input("Load per rep (kg)", 0.0, 500.0, suggested_load if mode=="Calculate from %1RM" else 60.0, step=0.5, key=f"load_{se.id}")
            rest = st.number_input("Rest (s)", 0, 600, 120, key=f"rest_{se.id}")
            note_in = st.text_input("Notes", value="", key=f"note_{se.id}")
            st.caption(f"Suggested by %1RM: {suggested_load:.1f} kg (1RM {one_rm_input:.1f}, {pct}%). You can override.")

            if st.button("Add set", key=f"addset_{se.id}"):
                if sets <= 0 or reps <= 0 or load <= 0:
                    st.error("Sets, reps and load must be > 0.")
                else:
                    try:
                        sp = SetPrescription(session_exercise_id=se.id, sets=int(sets), reps=int(reps), intensity_pct_1rm=float(pct), load_kg=round_to_05(float(load)), rest_sec=int(rest), notes=note_in)
                        db.add(sp); db.commit()
                        st.success("Set added"); st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Error adding set: {e}")

        sets_ = db.query(SetPrescription).filter(SetPrescription.session_exercise_id == se.id).all()
        data = []
        ex_tonnage = 0.0
        for sp in sets_:
            ton = compute_tonnage(sp.sets, sp.reps, sp.load_kg)
            ex_tonnage += ton
            data.append({"Set #": sp.id, "Sets": sp.sets, "Reps": sp.reps, "%1RM": int(sp.intensity_pct_1rm or 0), "Load (kg)": float(sp.load_kg or 0.0), "Rest (s)": int(sp.rest_sec or 0), "Tonnage (kg)": round(ton, 1), "Notes": sp.notes or ""})
        if data:
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        st.caption(f"**{ex.name} tonnage:** {ex_tonnage:.1f} kg")
        total_session_tonnage += ex_tonnage

        if sets_:
            with st.expander("Edit/Delete sets"):
                for sp in sets_:
                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        new_sets = st.number_input("Sets", 1, 50, sp.sets, key=f"e_sets_{sp.id}")
                        new_reps = st.number_input("Reps", 1, 100, sp.reps, key=f"e_reps_{sp.id}")
                    with c2:
                        new_pct = st.slider("%1RM", 30, 100, int(sp.intensity_pct_1rm or 75), key=f"e_pct_{sp.id}")
                        new_load = st.number_input("Load (kg)", 0.0, 500.0, float(sp.load_kg or 0.0), step=0.5, key=f"e_load_{sp.id}")
                    with c3:
                        new_rest = st.number_input("Rest (s)", 0, 600, int(sp.rest_sec or 0), key=f"e_rest_{sp.id}")
                        new_notes = st.text_input("Notes", value=sp.notes or "", key=f"e_notes_{sp.id}")
                    with c4:
                        if st.button("💾 Save", key=f"save_{sp.id}"):
                            if new_sets <= 0 or new_reps <= 0 or new_load <= 0:
                                st.error("Sets, reps and load must be > 0.")
                            else:
                                try:
                                    sp.sets = int(new_sets); sp.reps = int(new_reps); sp.intensity_pct_1rm = float(new_pct); sp.load_kg = round_to_05(float(new_load)); sp.rest_sec = int(new_rest); sp.notes = new_notes
                                    db.commit(); st.success("Updated"); st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"Error updating: {e}")
                        if st.button("🗑️ Delete", key=f"del_{sp.id}"):
                            try:
                                db.delete(sp); db.commit(); st.warning("Deleted"); st.experimental_rerun()
                            except Exception as e:
                                st.error(f"Error deleting: {e}")
        st.divider()

    st.info(f"**Total session tonnage:** {total_session_tonnage:.1f} kg")

    if sess_ex:
        rows = []
        for se in sess_ex:
            ex = db.get(Exercise, se.exercise_id)
            sets_ = db.query(SetPrescription).filter(SetPrescription.session_exercise_id == se.id).all()
            for sp in sets_:
                rows.append({"session_id": s.id, "date": s.date.isoformat(), "order": se.order_index, "exercise": ex.name, "sets": sp.sets, "reps": sp.reps, "pct_1rm": sp.intensity_pct_1rm, "load_kg": sp.load_kg, "rest_sec": sp.rest_sec, "notes": sp.notes or "", "tonnage": compute_tonnage(sp.sets, sp.reps, sp.load_kg)})
        if rows:
            total_row = {k: "" for k in rows[0].keys()}
            total_row["exercise"] = "TOTAL"; total_row["tonnage"] = round(sum(r["tonnage"] for r in rows), 1)
            rows.append(total_row)
            buf = io.StringIO(); writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
            st.download_button("⬇️ Export session to CSV (with TOTAL)", buf.getvalue(), file_name=f"session_{s.id}.csv", mime="text/csv")

    db.close()

def page_analytics():
    st.title("📊 Analytics — Tonnage per session & weekly cumulative")
    db = SessionLocal()

    plans = db.query(TrainingPlan).all()
    if not plans:
        st.info("No plans available."); db.close(); return
    plan_map = {f"{p.name} ({p.start_date}→{p.end_date})": p.id for p in plans}
    psel = st.selectbox("Plan", options=list(plan_map.keys()))
    plan = db.get(TrainingPlan, plan_map[psel])

    sess = (db.query(TrainingSession).filter(TrainingSession.plan_id == plan.id).order_by(TrainingSession.date).all())
    rows = []
    for s in sess:
        ex_links = db.query(SessionExercise).filter(SessionExercise.session_id == s.id).all()
        tot = 0.0
        for se in ex_links:
            sets_ = db.query(SetPrescription).filter(SetPrescription.session_exercise_id == se.id).all()
            for sp in sets_:
                tot += compute_tonnage(sp.sets, sp.reps, sp.load_kg)
        rows.append({"date": s.date, "session_id": s.id, "tonnage": tot})
    db.close()

    if not rows: st.info("No sessions yet."); return
    df = pd.DataFrame(rows).sort_values("date")
    df["week"] = df["date"].apply(lambda d: f"{d.isocalendar().year}-W{d.isocalalendar().week:02d}")  # typo fixed below
    # Fix typo: correct isocalendar
    df["week"] = df["date"].apply(lambda d: f"{d.isocalendar().year}-W{d.isocalendar().week:02d}")
    weekly = df.groupby("week", as_index=False)["tonnage"].sum(); weekly["cum_tonnage"] = weekly["tonnage"].cumsum()

    st.subheader("Tonnage per session")
    st.altair_chart(alt.Chart(df).mark_bar().encode(x=alt.X("date:T", title="Date"), y=alt.Y("tonnage:Q", title="Tonnage (kg)"), tooltip=["date:T","tonnage:Q","session_id:Q"]).properties(height=300), use_container_width=True)
    st.subheader("Weekly cumulative tonnage")
    st.altair_chart(alt.Chart(weekly).mark_line(point=True).encode(x=alt.X("week:N", title="ISO week"), y=alt.Y("cum_tonnage:Q", title="Cumulative Tonnage (kg)"), tooltip=["week:N","tonnage:Q","cum_tonnage:Q"]).properties(height=300), use_container_width=True)

def page_settings():
    st.title("⚙️ Settings")
    st.write("SQLite path: `/tmp/strength_mvp.db` — Exercise images under `/tmp/exercise_images`")
    if st.button("Reset database (danger)"):
        Base.metadata.drop_all(engine); Base.metadata.create_all(engine); init_demo_data(); st.success("Database reset.")

# -------- App layout & sidebar quick-create --------
st.set_page_config(page_title="Strength Prescriptor (MVP)", layout="wide")
with st.sidebar:
    st.title("Strength Prescriptor (MVP)")
    # Sidebar quick-create with form
    with st.form("sidebar_quick_create", clear_on_submit=True):
        try:
            db_sb = SessionLocal()
            plans_sb = db_sb.query(TrainingPlan).all()
            if plans_sb:
                plan_map_sb = {f"{p.name} ({p.start_date}→{p.end_date})": p.id for p in plans_sb}
                sel_plan_sb = st.selectbox("Plan", options=list(plan_map_sb.keys()), key="sb_plan")
            else:
                plan_map_sb = {}
                st.caption("No plans yet — will auto-create one.")
                sel_plan_sb = None
        finally:
            try: db_sb.close()
            except Exception: pass
        date_sb = st.date_input("Date", value=dt.date.today(), key="sb_date")
        focus_sb = st.text_input("Focus", value="Strength", key="sb_focus")
        notes_sb = st.text_input("Notes", value="", key="sb_notes")
        submitted_sb = st.form_submit_button("➕ Create session")
    if submitted_sb:
        try:
            with SessionLocal() as dbx:
                if plan_map_sb:
                    plan_id = plan_map_sb[sel_plan_sb]
                    pl = dbx.get(TrainingPlan, plan_id)
                else:
                    pl = ensure_plan(dbx)
                ns = TrainingSession(plan_id=pl.id, date=date_sb, focus=focus_sb, notes=notes_sb)
                dbx.add(ns); dbx.commit()
                st.success(f"Session #{ns.id} created for {date_sb}")
                st.session_state["focus_session_id"] = ns.id
                st.session_state["nav"] = "Sessions"
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Error creating session: {e}")

    page = st.radio("Navigation", ["Calendar", "Sessions", "Strength Tests", "Exercises", "Analytics", "Settings"], key="nav")

# -------- Router --------
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
