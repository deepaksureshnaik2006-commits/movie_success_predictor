import streamlit as st
import matplotlib.pyplot as plt
from predict import predict_movie

# Title
st.title("🎬 Movie Success Prediction AI")

st.write("Enter movie details to predict whether a movie will be a HIT or FLOP.")

# Movie name
movie_name = st.text_input("Enter Movie Name")

# Poster upload
poster = st.file_uploader("Upload Movie Poster", type=["jpg","png","jpeg"])

# Show poster in medium size
if poster:
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        st.image(poster, caption="Movie Poster", use_container_width=True)

# Movie inputs
budget = st.number_input("Movie Budget ($)")

genre = st.selectbox(
    "Genre",
    [
        "Action",
        "Thriller",
        "Drama",
        "Animation",
        "Musical",
        "Comedy",
        "Romance",
        "Horror",
        "Sci-Fi",
        "Adventure"
    ]
)

rating = st.slider("Expected IMDb Rating", 0.0, 10.0, 5.0)
cast = st.slider("Cast Popularity", 0, 100, 50)

# Predict button
if st.button("Predict"):

    # Input validation
    if movie_name.strip() == "" or budget == 0:
        st.warning("⚠️ Enter movie details before predicting.")

    else:
        rf_pred, dl_pred = predict_movie(budget, genre, rating, cast)

        st.subheader("🎬 Movie Prediction Result")

        if rf_pred == 1:
            st.success("HIT 🎉")
            hit_prob = 0.5 + (dl_pred / 2)
        else:
            st.error("FLOP ❌")
            hit_prob = dl_pred / 2

        flop_prob = 1 - hit_prob

        # Visualization
        st.subheader("📊 Prediction Visualization")

        labels = ["Hit Chance", "Flop Chance"]
        values = [hit_prob, flop_prob]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct="%1.1f%%")
        ax.set_title("Movie Success Prediction")

        st.pyplot(fig)