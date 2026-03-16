import streamlit as st
import matplotlib.pyplot as plt
from predict import predict_movie

# Title
st.title("🎬 Movie Success Prediction AI")

st.write("Enter movie details to predict whether a movie will be a HIT or FLOP.")

# Movie name
movie_name = st.text_input("Enter Movie Name")

# Director
director = st.text_input("Enter Director Name")

# Actors
actors = st.text_area("Enter Main Actor Names")

# Poster upload
poster = st.file_uploader("Upload Movie Poster", type=["jpg","png","jpeg"])

# Show poster centered
if poster:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(poster, caption="Movie Poster", use_container_width=True)

# Budget
budget = st.number_input(
    "Movie Budget ($)",
    min_value=0.0,
    step=1000000.0
)

# Genre
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

# IMDb rating
rating = st.slider("Expected IMDb Rating", 0.0, 10.0, 5.0)

# Cast popularity
cast = st.slider("Cast Popularity", 0, 100, 50)

# Predict
if st.button("Predict"):

    if movie_name.strip() == "" or budget == 0:
        st.warning("⚠️ Please enter all required movie details before predicting.")
    else:

        # Model prediction
        rf_pred, hit_prob = predict_movie(budget, genre, rating, cast)

        # Smooth probability adjustment using rating
        rating_factor = rating / 10
        hit_prob = (hit_prob * 0.6) + (rating_factor * 0.4)

        # Prevent extreme probabilities
        hit_prob = max(0.05, min(hit_prob, 0.95))
        flop_prob = 1 - hit_prob

        # Movie Details
        st.subheader("🎥 Movie Details")

        st.write("**Movie Name:**", movie_name)
        st.write("**Director:**", director if director else "Not provided")
        st.write("**Actors:**", actors if actors else "Not provided")
        st.write("**Genre:**", genre)
        st.write(f"**Budget:** ${budget:,.0f}")
        st.write("**Expected IMDb Rating:**", rating)

        # Final decision logic
        st.subheader("🎬 Movie Prediction Result")

        if rating >= 7:
            st.success("HIT 🎉")
        elif rating <= 4:
            st.error("FLOP ❌")
        else:
            if hit_prob >= 0.5:
                st.success("HIT 🎉")
            else:
                st.error("FLOP ❌")

        # Visualization
        st.subheader("📊 Prediction Visualization")

        labels = ["Hit Chance", "Flop Chance"]
        values = [hit_prob, flop_prob]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct="%1.1f%%")
        ax.set_title("Movie Success Probability")

        st.pyplot(fig)
