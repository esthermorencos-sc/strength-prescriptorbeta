
# Strength Prescriptor (MVP)

Minimal Streamlit app to prescribe **strength training**.

### Features
- **Strength Tests (1RM)** per client and exercise
- **Sessions**: choose exercise, sets, reps, %1RM and 1RM (autofilled from last test if available)
- **Suggested load** in kg (editable) based on %1RM × 1RM
- Persist **load_kg** and **intensity_pct_1rm**
- Compute **tonnage** per exercise and **total session tonnage** (sets × reps × kg)
- **No authentication** — database is created automatically on first run (SQLite)

---

## Run on Streamlit Cloud (recommended)

1. Push these files to a public GitHub repository:
    ```text
    app.py
    requirements.txt
    README.md
    ```
2. Go to https://share.streamlit.io (Streamlit Community Cloud) and sign in with GitHub.
3. Click **New app** and select your repository and branch.
4. Set **Main file path** to `app.py`.
5. Click **Deploy**.

The app will build and become available at a public URL like:
`https://<your-github-username>-<repo-name>.streamlit.app`

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app will open at: http://localhost:8501

---

## Notes

- The app seeds a demo coach, client, exercises, and a 4-week training plan on first run.
- The SQLite database file `strength_mvp.db` is created in the working directory.
- For production use, consider: Postgres, authentication (SSO), role-based access, backups, and GDPR compliance.
