


# ====================================================
# 🎵 MUSIC REPLAY & SEARCH SYSTEM STREAMLIT APP
# ====================================================

import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

# ====================================================
# 🧠 LOAD SAVED MODEL, SCALER, AND DATA
# ====================================================

# Get working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load trained model & scaler
model_path = f"{working_dir}/music_recommendation_prediction_model.sav"
scaler_path = f"{working_dir}/replay_scaler.sav"

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))

# Load dataset (make sure all decades CSVs are in the same folder)
csv_files = [
    "dataset-of-60s.csv",
    "dataset-of-70s.csv",
    "dataset-of-80s.csv",
    "dataset-of-90s.csv",
    "dataset-of-00s.csv",
    "dataset-of-10s.csv"
]
dfs = [pd.read_csv(f"{working_dir}/{f}") for f in csv_files]
df = pd.concat(dfs, ignore_index=True)

# Add necessary simulated columns if not present
if "user_id" not in df.columns:
    np.random.seed(42)
    df["user_id"] = np.random.randint(1, 1001, size=df.shape[0])
if "plays_in_last_30_days" not in df.columns:
    df["plays_in_last_30_days"] = np.random.poisson(2, size=df.shape[0])
if "replay_probability" not in df.columns:
    df["replay_probability"] = np.random.uniform(0, 1, size=df.shape[0])

# ====================================================
# 🎛️ STREAMLIT PAGE CONFIGURATION
# ====================================================
st.set_page_config(page_title="🎧 Music Predictive System", layout="wide", page_icon="🎵")
st.title("🎧 Music Predictive System – Top Songs & Song Search")

# ====================================================
# 🧠 STEP 9: Top 10 Most Played Songs (Auto for User 42)
# ====================================================

example_user = 42
top_n = 10

st.markdown(f"### 🎧 Top {top_n} Most Played Songs for User {example_user} (by plays_in_last_30_days)")

# Filter and sort songs for the user
user_songs = df[df["user_id"] == example_user]
top_songs = user_songs.sort_values("plays_in_last_30_days", ascending=False).head(top_n)

if not top_songs.empty:
    st.dataframe(
        top_songs[["track", "artist", "plays_in_last_30_days", "replay_probability"]],
        use_container_width=True,
        hide_index=True
    )
else:
    st.warning("⚠️ No songs found for this user.")

# ====================================================
# 🔍 SONG SEARCH SECTION
# ====================================================

st.markdown("### 🔍 Search for Any Song in the Dataset")
search_song = st.text_input("Enter a song name:")

if search_song:
    result = df[df["track"].str.contains(search_song, case=False, na=False)]

    if not result.empty:
        st.success(f"🎵 Found {len(result)} result(s) for '{search_song}':")
        st.dataframe(
            result[["track", "artist", "plays_in_last_30_days", "replay_probability"]]
            .drop_duplicates()
            .sort_values("plays_in_last_30_days", ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.error(f"❌ '{search_song}' not present in the dataset.")
else:
    st.info("Type a song name above to search.")

# ====================================================
# 🧠 STEP 10: END MESSAGE
# ====================================================
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit, Scikit-learn, and your trained Music Recommendation Model.")
