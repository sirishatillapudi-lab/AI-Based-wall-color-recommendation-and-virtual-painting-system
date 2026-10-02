import os
from io import BytesIO

import streamlit as st
import pandas as pd
from PIL import Image

from models.virtual_paint import (
    detect_surfaces,
    create_surface_visualization,
    paint_detected_surfaces
)


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="AI Wall Colour Recommendation",
    page_icon="🎨",
    layout="wide"
)


st.title(
    "🎨 AI-Based Wall & Ceiling Colour Recommendation System"
)

st.write(
    "AI detects visible walls and ceiling separately, "
    "recommends suitable colours, and creates a "
    "virtual painting preview."
)


# =========================================================
# LOAD COLOUR DATABASE
# =========================================================

COLOR_FILE = "data/colors.csv"


if not os.path.exists(COLOR_FILE):

    st.error(
        "❌ colors.csv not found."
    )

    st.stop()


color_df = pd.read_csv(
    COLOR_FILE
)


required_columns = [
    "No",
    "Color",
    "ShadeCode",
    "HEX",
    "Category"
]


if not all(
    column in color_df.columns
    for column in required_columns
):

    st.error(
        "❌ colors.csv must contain: "
        "No, Color, ShadeCode, HEX, Category"
    )

    st.stop()


color_df["No"] = (
    color_df["No"]
    .astype(str)
    .str.zfill(3)
)


color_df["ShadeCode"] = (
    color_df["ShadeCode"]
    .astype(str)
)


color_df = color_df.drop_duplicates(
    subset=["Color"]
).reset_index(drop=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "🏠 Room Settings"
)


st.sidebar.success(
    f"🎨 {len(color_df)} colours loaded"
)


room_type = st.sidebar.selectbox(
    "Room Type",
    [
        "Living Room",
        "Bedroom",
        "Kitchen",
        "Dining Room",
        "Study Room",
        "Kids Room"
    ]
)


category = st.sidebar.selectbox(
    "Colour Category",
    [
        "All",
        "Neutral",
        "Warm",
        "Cool",
        "Pastel",
        "Bright",
        "Dark"
    ]
)


if category == "All":

    filtered_colors = color_df.copy()

else:

    filtered_colors = color_df[
        color_df["Category"].str.lower()
        == category.lower()
    ]


available_colors = (
    filtered_colors["Color"].tolist()
)


if len(available_colors) == 0:

    st.error(
        "❌ No colours available."
    )

    st.stop()


# =========================================================
# IMAGE INPUT
# =========================================================

st.header(
    "📷 Choose Room Image"
)


input_method = st.radio(
    "Select image source:",
    [
        "📁 Upload from Device",
        "📸 Take Photo with Camera"
    ],
    horizontal=True
)


uploaded_file = None


if input_method == "📁 Upload from Device":

    uploaded_file = st.file_uploader(
        "Choose a JPG or PNG image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

else:

    uploaded_file = st.camera_input(
        "Take a photo of your room"
    )


if uploaded_file is None:

    st.info(
        "👆 Upload an image or take a room photo."
    )

    st.stop()


image = Image.open(
    uploaded_file
).convert("RGB")


# =========================================================
# ORIGINAL IMAGE
# =========================================================

st.subheader(
    "📷 Original Room"
)


st.image(
    image,
    use_container_width=True
)


# =========================================================
# AI DETECTION
# =========================================================

with st.spinner(
    "🤖 AI is detecting walls and ceiling..."
):

    surfaces = detect_surfaces(
        image
    )


ceiling_surfaces = [
    surface
    for surface in surfaces
    if surface["type"] == "ceiling"
]


wall_surfaces = [
    surface
    for surface in surfaces
    if surface["type"] == "wall"
]


ceiling_count = len(
    ceiling_surfaces
)


wall_count = len(
    wall_surfaces
)


# =========================================================
# DETECTION RESULT
# =========================================================

st.header(
    "🔎 AI Surface Detection"
)


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "🧱 Walls Detected",
        wall_count
    )


with col2:

    st.metric(
        "⬆️ Ceiling Detected",
        "YES"
        if ceiling_count > 0
        else "NO"
    )


if (
    wall_count == 0
    and
    ceiling_count == 0
):

    st.error(
        "❌ AI could not detect a wall or ceiling."
    )

    st.stop()


# =========================================================
# DETECTION VISUALIZATION
# =========================================================

visualization = (
    create_surface_visualization(
        image,
        surfaces
    )
)


st.image(
    visualization,
    caption="AI detected ceiling and walls",
    use_container_width=True
)


# =========================================================
# ROOM RECOMMENDATIONS
# =========================================================

room_recommendations = {

    "Living Room": [
        "Beige",
        "Cream",
        "Sage Green",
        "Sky Blue"
    ],

    "Bedroom": [
        "Lavender",
        "Powder Blue",
        "Sage Green",
        "Blush"
    ],

    "Kitchen": [
        "Cream",
        "White",
        "Peach",
        "Light Grey"
    ],

    "Dining Room": [
        "Terracotta",
        "Beige",
        "Mustard",
        "Cream"
    ],

    "Study Room": [
        "Light Grey",
        "Sage Green",
        "Sky Blue",
        "Ivory"
    ],

    "Kids Room": [
        "Baby Blue",
        "Pink",
        "Mint",
        "Lavender"
    ]
}


recommendations = (
    room_recommendations[
        room_type
    ]
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_color_row(color_name):

    row = color_df[
        color_df["Color"].str.lower()
        == color_name.lower()
    ]

    if len(row) == 0:

        return None

    return row.iloc[0]


def get_safe_color(
    preferred_color,
    fallback_index=0
):

    if preferred_color in available_colors:

        return preferred_color

    return available_colors[
        fallback_index
        % len(available_colors)
    ]


def hex_to_rgb(hex_value):

    hex_value = (
        str(hex_value)
        .replace("#", "")
        .strip()
    )

    return tuple(
        int(
            hex_value[i:i + 2],
            16
        )
        for i in (0, 2, 4)
    )


# =========================================================
# AI RECOMMENDED COLOURS
# =========================================================

st.header(
    "🤖 AI Recommended Colours"
)

st.write(
    f"AI recommendations for your {room_type} "
    "are shown below."
)


# ---------------------------------------------------------
# RECOMMENDATION REASONS
# ---------------------------------------------------------

recommendation_reasons = {

    "Living Room":
        "A warm or neutral shade can create a comfortable "
        "and welcoming living-room appearance.",

    "Bedroom":
        "Soft and gentle shades can create a calm "
        "and relaxing bedroom appearance.",

    "Kitchen":
        "Light and clean-looking shades can create "
        "a fresh kitchen appearance.",

    "Dining Room":
        "Warm and earthy shades can create a "
        "comfortable dining-room appearance.",

    "Study Room":
        "Soft neutral or cool shades can create a "
        "comfortable study environment.",

    "Kids Room":
        "Soft and cheerful shades can create a "
        "pleasant and playful room appearance."
}


# ---------------------------------------------------------
# AI SURFACE COLOUR LIST
# ---------------------------------------------------------

ai_surface_names = []


# =========================================================
# CEILING RECOMMENDATION
# =========================================================

if ceiling_count > 0:

    ceiling_recommended = get_safe_color(
        "White"
    )

    ai_surface_names.append(
        ceiling_recommended
    )

    ceiling_row = get_color_row(
        ceiling_recommended
    )

    st.subheader(
        "⬆️ Ceiling Recommendation"
    )

    st.success(
        f"🎨 **{ceiling_recommended}**"
    )

    st.write(
        f"🏷️ Shade Code: "
        f"**{ceiling_row['ShadeCode']}**"
    )

    st.write(
        "💡 **Why AI selected this:** "
        "A light ceiling shade can help maintain "
        "a bright appearance in the room."
    )


# =========================================================
# WALL RECOMMENDATIONS
# =========================================================

st.subheader(
    "🧱 Wall Recommendations"
)


for i in range(wall_count):

    recommended = recommendations[
        i % len(recommendations)
    ]

    recommended = get_safe_color(
        recommended,
        i
    )

    ai_surface_names.append(
        recommended
    )

    row = get_color_row(
        recommended
    )

    st.info(
        f"🧱 **WALL {i + 1}**\n\n"
        f"🎨 Recommended Colour: "
        f"**{recommended}**\n\n"
        f"🏷️ Shade Code: "
        f"**{row['ShadeCode']}**"
    )

    st.write(
        "💡 **Why AI selected this:** "
        f"{recommendation_reasons[room_type]}"
    )


# =========================================================
# RECOMMENDATION SUMMARY
# =========================================================

st.subheader(
    "📋 AI Recommendation Summary"
)

summary_data = []

surface_number = 1

if ceiling_count > 0:

    row = get_color_row(
        ai_surface_names[0]
    )

    summary_data.append({
        "Surface": "Ceiling",
        "Recommended Colour":
            ai_surface_names[0],
        "Shade Code":
            row["ShadeCode"]
    })

    surface_number = 1


for i in range(wall_count):

    index = (
        surface_number + i
        if ceiling_count > 0
        else i
    )

    row = get_color_row(
        ai_surface_names[index]
    )

    summary_data.append({
        "Surface":
            f"Wall {i + 1}",
        "Recommended Colour":
            ai_surface_names[index],
        "Shade Code":
            row["ShadeCode"]
    })


summary_df = pd.DataFrame(
    summary_data
)

st.dataframe(
    summary_df,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# AI RECOMMENDED VIRTUAL PAINTING
# =========================================================

st.header(
    "✨ AI Recommended Virtual Painting"
)


ai_rgb_colors = []


for name in ai_surface_names:

    row = get_color_row(name)

    rgb = hex_to_rgb(
        row["HEX"]
    )

    ai_rgb_colors.append(
        rgb
    )


with st.spinner(
    "🎨 Painting ceiling and walls..."
):

    ai_result = paint_detected_surfaces(
        image,
        surfaces,
        ai_rgb_colors
    )


st.image(
    ai_result,
    caption=(
        "AI Recommended Ceiling + Wall Colours"
    ),
    use_container_width=True
)


st.success(
    "✅ AI recommended colours have been "
    "applied to the detected ceiling and walls."
)


# =========================================================
# USER COLOUR SELECTION
# =========================================================

st.header(
    "🎨 Choose Your Own Colours"
)


st.write(
    "You can change the ceiling colour and "
    "each wall colour separately."
)


user_surface_names = []


# =========================================================
# CEILING SELECTION
# =========================================================

if ceiling_count > 0:

    st.subheader(
        "⬆️ Ceiling Colour"
    )

    default_ceiling = get_safe_color(
        "White"
    )

    selected_ceiling = st.selectbox(
        "Choose colour for Ceiling",
        available_colors,
        index=available_colors.index(
            default_ceiling
        ),
        key="ceiling_colour"
    )

    user_surface_names.append(
        selected_ceiling
    )

    row = get_color_row(
        selected_ceiling
    )

    st.success(
        f"⬆️ **CEILING** → "
        f"🎨 **{selected_ceiling}**  |  "
        f"🏷️ Shade Code: "
        f"**{row['ShadeCode']}**"
    )


# =========================================================
# WALL SELECTION
# =========================================================

st.subheader(
    "🧱 Wall Colours"
)


for i in range(wall_count):

    default_wall = (
        recommendations[
            i % len(recommendations)
        ]
    )

    default_wall = get_safe_color(
        default_wall,
        i
    )

    selected_wall = st.selectbox(
        f"Choose colour for Wall {i + 1}",
        available_colors,
        index=available_colors.index(
            default_wall
        ),
        key=f"wall_colour_{i}"
    )

    user_surface_names.append(
        selected_wall
    )

    row = get_color_row(
        selected_wall
    )

    st.success(
        f"🧱 **WALL {i + 1}** → "
        f"🎨 **{selected_wall}**  |  "
        f"🏷️ Shade Code: "
        f"**{row['ShadeCode']}**"
    )


# =========================================================
# APPLY USER COLOURS
# =========================================================

if st.button(
    "🎨 Apply My Colours",
    type="primary"
):

    user_rgb_colors = []

    for name in user_surface_names:

        row = get_color_row(name)

        rgb = hex_to_rgb(
            row["HEX"]
        )

        user_rgb_colors.append(
            rgb
        )

    with st.spinner(
        "🎨 Creating your ceiling + wall preview..."
    ):

        user_result = paint_detected_surfaces(
            image,
            surfaces,
            user_rgb_colors
        )

    st.header(
        "✨ Your Colour Preview"
    )

    st.image(
        user_result,
        caption=(
            "Your Selected Ceiling + Wall Colours"
        ),
        use_container_width=True
    )

    st.success(
        "✅ Your selected ceiling and wall "
        "colours were applied successfully."
    )

    # -----------------------------------------------------
    # DOWNLOAD
    # -----------------------------------------------------

    result_image = Image.fromarray(
        user_result
    )

    buffer = BytesIO()

    result_image.save(
        buffer,
        format="PNG"
    )

    st.download_button(
        "⬇️ Download Painted Room",
        data=buffer.getvalue(),
        file_name=(
            "wall_and_ceiling_colour_result.png"
        ),
        mime="image/png"
    )


# =========================================================
# COLOUR CATALOGUE
# =========================================================

st.header(
    "🎨 Colour Catalogue"
)


st.write(
    f"Total Colours: **{len(color_df)}**"
)


catalogue_columns = st.columns(5)


for index, row in color_df.iterrows():

    with catalogue_columns[
        index % 5
    ]:

        st.markdown(
            f"""
            <div style="
                background-color:{row['HEX']};
                height:60px;
                border-radius:10px;
                border:1px solid #777;
            ">
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write(
            f"🎨 {row['Color']}"
        )

        st.caption(
            f"🏷️ {row['ShadeCode']}"
        )