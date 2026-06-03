import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("rf_model.joblib")
model_features = joblib.load("model_columns.joblib")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("cleaned_df.csv")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background-color: #F8FAFC;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #E2E8F0;
}

/* Buttons */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#1D4ED8,#1E3A8A);
    color: white;
    font-size: 22px;
    font-weight: 700;
    border-radius: 12px;
    border: none;
    padding: 14px;
}

.stDownloadButton > button {
    width: 100%;
    background: #16A34A;
    color: white;
    font-size: 22px;
    font-weight: 700;
    border-radius: 12px;
    border: none;
    padding: 14px;
}

/* Metric Card */
[data-testid="stMetric"] {
    background: linear-gradient(135deg,#1D4ED8,#1E3A8A);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 10px 25px rgba(0,0,0,0.15);
}

[data-testid="stMetricLabel"] {
    color: white !important;
    font-size: 28px !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 72px !important;
    font-weight: 800 !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<h1 style='text-align:center;
font-size:70px;
font-weight:800;
color:#1D4ED8;
margin-bottom:0px;'>
🏠 House Price Prediction System
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<p style='text-align:center;
font-size:30px;
color:#475569;
margin-top:0px;'>
Premium Real Estate ML Estimator
</p>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.title("🏠 Web Info")

    st.markdown("""
    ### Instructions

    • Select location

    • Enter property details

    • Click Predict Price

    • Download Report
    """)

    st.markdown("---")

    st.markdown("Built using Machine Learning")

# ---------------- LOCATIONS ----------------
locations = [
    col.replace("location_", "")
    for col in model_features
    if col.startswith("location_")
]

# ---------------- INPUT SECTION ----------------
col1, col2 = st.columns(2)

with col1:

    location = st.selectbox(
        "📍 Location",
        sorted(locations)
    )

    sqft = st.number_input(
        "📐 Total Square Feet",
        min_value=300,
        value=1200
    )

with col2:

    bath = st.selectbox(
        "🛁 Bathrooms",
        sorted(df["bath"].unique())
    )

    bhk = st.selectbox(
        "🏠 BHK",
        sorted(df["bhk"].unique())
    )

# ---------------- PREPARE INPUT ----------------
def prepare_input():

    input_dict = {
        col: 0 for col in model_features
    }

    input_dict["total_sqft"] = sqft
    input_dict["bath"] = bath
    input_dict["bhk"] = bhk

    loc_col = f"location_{location}"

    if loc_col in input_dict:
        input_dict[loc_col] = 1

    return pd.DataFrame([input_dict])

# ---------------- PREDICTION ----------------
if st.button("💰 Predict Price"):

    try:

        # Predict
        input_df = prepare_input()

        prediction = model.predict(input_df)

        price = round(float(prediction[0]), 2)

        # ---------------- RESULT CARD ----------------
        st.markdown("## 🏠 Estimated Market Value")

        st.metric(
            label="Predicted House Price",
            value=f"₹ {price*100000:,.0f}"
        )

        # ---------------- SIMILAR PROPERTIES ----------------
        st.subheader("💡 Price Insight")

        similar_props = df[
            (df["total_sqft"].between(sqft * 0.8, sqft * 1.2)) &
            (df["bhk"] == bhk) &
            (df["bath"].between(bath - 1, bath + 1))
        ]

        if similar_props.empty or len(similar_props) < 3:

            st.warning(
                "⚠️ Not enough similar properties found."
            )

        else:

            # ---------------- AVERAGE PRICE ----------------
            avg_price = round(
                float(similar_props["price"].mean()),
                2
            )

            # ---------------- DIFFERENCE ----------------
            diff_percent = (
                ((price - avg_price) / avg_price) * 100
            )

            # ---------------- INSIGHT ----------------
            if diff_percent < -10:

                st.success(
                    f"🟢 Underpriced by {abs(diff_percent):.1f}% compared to market average"
                )

            elif -10 <= diff_percent <= 10:

                st.info(
                    "🟡 Fairly priced according to market trend"
                )

            else:

                st.error(
                    f"🔴 Overpriced by {diff_percent:.1f}% compared to market average"
                )

            st.caption(
                f"📊 Based on {len(similar_props)} similar properties"
            )

            # ---------------- GRAPH ----------------
            st.subheader("📊 Your Price vs Similar Properties")

            fig, ax = plt.subplots(figsize=(11, 5))

            # Histogram
            ax.hist(
                similar_props["price"],
                bins=12,
                alpha=0.6,
                edgecolor='#1E3A8A',
                linewidth=1.5
            )

            # Predicted Price
            ax.axvline(
                price,
                color='red',
                linestyle='--',
                linewidth=3,
                label=f'Predicted Price = ₹ {price:.2f} Lakhs'
            )

            # Average Price
            ax.axvline(
                avg_price,
                color='green',
                linestyle='-.',
                linewidth=3,
                label=f'Average Market Price = ₹ {avg_price:.2f} Lakhs'
            )

            # Graph Title
            ax.set_title(
                "Your Price vs Similar Properties",
                fontsize=22,
                fontweight='bold'
            )

            # Labels
            ax.set_xlabel(
                "Price (Lakhs)",
                fontsize=15,
                fontweight='bold'
            )

            ax.set_ylabel(
                "Number of Properties",
                fontsize=15,
                fontweight='bold'
            )

            # Grid
            ax.grid(
                True,
                linestyle='--',
                alpha=0.35
            )

            # Legend
            ax.legend(
                fontsize=11
            )

            plt.tight_layout()

            # Show graph
            st.pyplot(fig, use_container_width=True)

            # Save graph
            graph_path = "market_analysis.png"

            fig.savefig(
                graph_path,
                bbox_inches='tight'
            )

            # ---------------- PDF REPORT ----------------
            pdf_path = "House_Price_Report.pdf"

            doc = SimpleDocTemplate(
                pdf_path,
                pagesize=letter,
                topMargin=20,
                bottomMargin=20,
                leftMargin=30,
                rightMargin=30
            )

            styles = getSampleStyleSheet()

            elements = []

            # ---------------- TITLE ----------------
            title = Paragraph(
                "<font size=20><b>House Price Prediction Report</b></font>",
                styles['Title']
            )

            elements.append(title)

            elements.append(
                Spacer(1, 0.15 * inch)
            )

            # ---------------- TABLE ----------------
            data = [
                ["Property Detail", "Value"],
                ["Location", location],
                ["Total Sqft", str(sqft)],
                ["Bathrooms", str(bath)],
                ["BHK", str(bhk)],
                ["Predicted Price", f"₹ {price*100000:,.0f}"],
                ["Average Market Price", f"₹ {avg_price*100000:,.0f}"],
                ["Similar Properties", str(len(similar_props))]
            ]

            table = Table(
                data,
                colWidths=[220, 220]
            )

            table.setStyle(TableStyle([

                (
                    'BACKGROUND',
                    (0,0),
                    (-1,0),
                    colors.HexColor("#1E3A8A")
                ),

                (
                    'TEXTCOLOR',
                    (0,0),
                    (-1,0),
                    colors.white
                ),

                (
                    'FONTNAME',
                    (0,0),
                    (-1,0),
                    'Helvetica-Bold'
                ),

                (
                    'FONTSIZE',
                    (0,0),
                    (-1,-1),
                    9
                ),

                (
                    'GRID',
                    (0,0),
                    (-1,-1),
                    1,
                    colors.grey
                ),

                (
                    'BACKGROUND',
                    (0,1),
                    (-1,-1),
                    colors.whitesmoke
                )

            ]))

            elements.append(table)

            elements.append(
                Spacer(1, 0.15 * inch)
            )

            # ---------------- INSIGHT ----------------
            insight = Paragraph(
                f"""
                <b>Market Insight</b><br/>
                Predicted Property Value:
                <b>₹ {price*100000:,.0f}</b><br/>
                Average Market Value:
                <b>₹ {avg_price*100000:,.0f}</b><br/>
                Difference:
                <b>{diff_percent:.1f}%</b>
                """,
                styles['BodyText']
            )

            elements.append(insight)

            elements.append(
                Spacer(1, 0.15 * inch)
            )

            # ---------------- GRAPH ----------------
            report_graph = Image(
                graph_path,
                width=6.7 * inch,
                height=3.2 * inch
            )

            elements.append(report_graph)

            elements.append(
                Spacer(1, 0.1 * inch)
            )

            # ---------------- FOOTER ----------------
            footer = Paragraph(
                "Generated using House Price Prediction ML Model",
                styles['Italic']
            )

            elements.append(footer)

            # Build PDF
            doc.build(elements)

            # ---------------- DOWNLOAD BUTTON ----------------
            with open(pdf_path, "rb") as pdf_file:

                st.download_button(
                    label="📄 Download Predicted Report",
                    data=pdf_file,
                    file_name="House_Price_Report.pdf",
                    mime="application/pdf"
                )

    except Exception as e:

        st.error(f"Error: {e}")

# ---------------- FOOTER ----------------
st.markdown("---")

