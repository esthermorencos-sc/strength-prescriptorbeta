# Strength Prescriptor (MVP)

Minimal Streamlit app to prescribe **strength training**.

### Features
- **Strength Tests (1RM)** per client and exercise
- **Sessions**: choose exercise, sets, reps, %1RM and 1RM (autofilled from last test if available)
- **Suggested load** in kg (editable) based on %1RM × 1RM
- **Edit/Delete** sets
- **Per-exercise tonnage** and **total session tonnage**
- **Export CSV** (includes a final TOTAL tonnage row)
- **No authentication** — database is created automatically on first run (SQLite)

## Run on Streamlit Cloud (recommended)
1. Push these files to a public GitHub repository:
   - `app.py`
   - `requirements.txt`
   - `README.md`
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app** and select your repository and branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
The app will open at: http://localhost:8501
