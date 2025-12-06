import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Prediksi Banjir",
    page_icon="🌊",
    layout="wide"
)

# CSS untuk styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .category-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.75rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #155a8a;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 2rem;
        border-radius: 10px;
        border: 2px solid #1f77b4;
        text-align: center;
        margin-top: 2rem;
    }
    .feature-description {
        font-size: 0.85rem;
        color: rgba(0, 0, 0, 0.7);
        margin-top: 0.25rem;
        line-height: 1.4;
        font-style: italic;
    }
    </style>
""", unsafe_allow_html=True)

# Definisi kategori dan fitur
CATEGORIES = {
    "Lingkungan & Cuaca": [
        "MonsoonIntensity", "ClimateChange", "CoastalVulnerability", 
        "Landslides", "Watersheds", "TopographyDrainage"
    ],
    "Kerusakan Lingkungan": [
        "Deforestation", "Siltation", "WetlandLoss", "AgriculturalPractices"
    ],
    "Infrastruktur": [
        "DamsQuality", "DrainageSystems", "DeterioratingInfrastructure"
    ],
    "Urbanisasi & Aktivitas Manusia": [
        "Urbanization", "Encroachments", "PopulationScore", "InadequatePlanning"
    ],
    "Tata Kelola & Kebijakan": [
        "RiverManagement", "IneffectiveDisasterPreparedness", "PoliticalFactors"
    ]
}

# Deskripsi fitur dalam Bahasa Indonesia
FEATURE_DESCRIPTIONS = {
    "MonsoonIntensity": "Tingkat Volume hujan (0-10).",
    "TopographyDrainage": "Tingkat buruknya drainase (semakin tinggi nilai semakin buruk) (0-10).",
    "RiverManagement": "Kualitas buruknya dan ketidakefektivitas praktik pengelolaan sungai (semakin tinggi nilai semakin baik) (0-10).",
    "Deforestation": "Tingkat deforestasi di wilayah tersebut (semakin tinggi nilai semakin sering terjadi)(0-10).",
    "Urbanization": "Tingkat urbanisasi di wilayah (0-10).",
    "ClimateChange": "Dampak perubahan iklim terhadap wilayah (0-10).",
    "DamsQuality": "Tingkat buruknya kualitas dan status pemeliharaan bendungan (semakin tinggi nilai semakin buruk) (0-10).",
    "Siltation": "Tingkat buruknya sedimentasi di sungai dan waduk (semakin tinggi nilai semakin buruk) (0-10).",
    "AgriculturalPractices": "Jumlah wilayah pertanian yang dibabat (semakin tinggi nilai semakin banyak) (0-10).",
    "Encroachments": "Tingkat buruknya penyerapan pada dataran banjir dan jalur air alami (0-10).",
    "IneffectiveDisasterPreparedness": "Nilai buruknya Mitigasi rencana darurat, sistem peringatan, dan simulasi meningkatkan dampak banjir (semakin tinggi nilai semakin buruk) (0-10).",
    "DrainageSystems": "Buruknya Sistem drainase (semakin tinggi nilai semakin buruk) (0-10).",
    "CoastalVulnerability": "Jumlah area pesisir yang rentan (semakin tinggi nilai semakin banyak) (0-10).",
    "Landslides": "Jumlah area rentan (lahan longsor) (semakin tinggi nilai semakin banyak) (0-10).",
    "Watersheds": "Jumlah area aliran sungai dan sejenisnya (semakin tinggi nilai semakin banyak) (0-10).",
    "DeterioratingInfrastructure": "Infrastruktur yang rusak (semakin tinggi nilai semakin banyak) (0-10).",
    "PopulationScore": "Jumlah populasi yang dapat terpengaruh (semakin tinggi nilai semakin banyak) (0-10).",
    "WetlandLoss": "Jumlah kehilangan lahan basah yang dapat menyerap (semakin tinggi nilai semakin banyak) (0-10).",
    "InadequatePlanning": "Buruknya Perencanaan tata kota (semakin tinggi nilai semakin buruk) (0-10).",
    "PoliticalFactors": "Buruknya Faktor politik mendukung pengelolaan banjir ?(semakin tinggi nilai semakin buruk) (0-10)."
}

# Template data untuk lokasi berbeda
TEMPLATES = {
    "Sumatera": {
        "MonsoonIntensity": 9.2,
        "TopographyDrainage": 7.8,
        "RiverManagement": 8.5,
        "Deforestation": 9.0,
        "Urbanization": 7.5,
        "ClimateChange": 8.8,
        "DamsQuality": 8.2,
        "Siltation": 8.9,
        "AgriculturalPractices": 7.8,
        "Encroachments": 8.5,
        "IneffectiveDisasterPreparedness": 9.1,
        "DrainageSystems": 8.7,
        "CoastalVulnerability": 7.9,
        "Landslides": 8.6,
        "Watersheds": 8.4,
        "DeterioratingInfrastructure": 8.8,
        "PopulationScore": 8.3,
        "WetlandLoss": 8.9,
        "InadequatePlanning": 9.0,
        "PoliticalFactors": 7.7
    },
    "Belanda": {
        "MonsoonIntensity": 2.1,
        "TopographyDrainage": 1.5,
        "RiverManagement": 1.2,
        "Deforestation": 1.8,
        "Urbanization": 3.2,
        "ClimateChange": 2.5,
        "DamsQuality": 1.3,
        "Siltation": 1.7,
        "AgriculturalPractices": 2.0,
        "Encroachments": 1.4,
        "IneffectiveDisasterPreparedness": 1.1,
        "DrainageSystems": 1.0,
        "CoastalVulnerability": 2.8,
        "Landslides": 1.2,
        "Watersheds": 1.6,
        "DeterioratingInfrastructure": 1.5,
        "PopulationScore": 3.5,
        "WetlandLoss": 1.9,
        "InadequatePlanning": 1.3,
        "PoliticalFactors": 1.8
    }
}

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False
if 'show_review' not in st.session_state:
    st.session_state.show_review = False

def load_model():
    """Load model prediksi"""
    try:
        with open('best_flood_prediction_model.pkl', 'rb') as f:
            return pickle.load(f)

    except ModuleNotFoundError as e:
        st.error(f"❗ Module yang hilang saat load model: {e}")
        raise

    except Exception as e:
        st.error(f"❗ Error lain saat load model: {e}")
        raise

def save_form_data(category, data):
    """Simpan data form untuk kategori tertentu"""
    st.session_state.form_data[category] = data

def load_template(template_name):
    """Load template data untuk lokasi tertentu"""
    template_data = TEMPLATES.get(template_name, {})
    
    # Distribusikan data template ke kategori yang sesuai
    st.session_state.form_data = {}
    for category, features in CATEGORIES.items():
        category_data = {}
        for feature in features:
            if feature in template_data:
                category_data[feature] = template_data[feature]
        if category_data:
            st.session_state.form_data[category] = category_data
    
    # Reset ke langkah pertama dan show review
    st.session_state.current_step = 0
    st.session_state.show_review = True
    st.session_state.prediction_made = False

def get_risk_level(probability):
    """Tentukan level risiko berdasarkan probabilitas"""
    if probability < 0.35:
        return "RENDAH", "🟢", "#28a745"
    elif probability < 0.50:
        return "SEDANG", "🟡", "#ffc107"
    elif probability < 0.65:
        return "TINGGI", "🟠", "#fd7e14"
    else:
        return "SANGAT TINGGI", "🔴", "#dc3545"

def main():
    # Header
    st.markdown("<h1 class='main-header'>🌊 Sistem Prediksi Banjir Sumatera</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Template Preset Section
    st.markdown("### 📍 Gunakan Template Lokasi")
    col_template1, col_template2, col_template3 = st.columns([1, 1, 2])
    
    with col_template1:
        if st.button("🇮🇩 Sumatera (High Risk)", use_container_width=True, type="secondary"):
            load_template("Sumatera")
            st.rerun()
    
    with col_template2:
        if st.button("🇳🇱 Belanda (Low Risk)", use_container_width=True, type="secondary"):
            load_template("Belanda")
            st.rerun()
    
    with col_template3:
        st.info("💡 Klik template untuk auto-fill data, atau isi manual di bawah")
    
    st.markdown("---")
    
    # Progress bar
    category_list = list(CATEGORIES.keys())
    total_steps = len(category_list)
    
    # Tambahkan step untuk review
    if st.session_state.show_review:
        progress = 1.0
        st.progress(progress)
        st.markdown(f"**Review Data - Siap untuk Prediksi**")
    else:
        progress = st.session_state.current_step / total_steps
        st.progress(progress)
        st.markdown(f"**Langkah {st.session_state.current_step + 1} dari {total_steps}**")
    
    # Tampilkan form jika belum selesai
    if st.session_state.current_step < total_steps and not st.session_state.prediction_made and not st.session_state.show_review:
        current_category = category_list[st.session_state.current_step]
        features = CATEGORIES[current_category]
        
        st.markdown(f"<h2 class='category-header'>📋 {current_category}</h2>", unsafe_allow_html=True)
        
        # Buat form untuk kategori saat ini
        with st.form(key=f"form_{st.session_state.current_step}"):
            col1, col2 = st.columns(2)
            form_values = {}
            
            for idx, feature in enumerate(features):
                col = col1 if idx % 2 == 0 else col2
                with col:
                    # Ambil nilai sebelumnya jika ada
                    default_value = st.session_state.form_data.get(current_category, {}).get(feature, 5.0)
                    
                    # Label fitur
                    st.markdown(f"**{feature}**")
                    
                    # Deskripsi fitur dengan styling
                    description = FEATURE_DESCRIPTIONS.get(feature, "")
                    if description:
                        st.markdown(f"<p class='feature-description'>{description}</p>", unsafe_allow_html=True)
                    
                    # Slider
                    form_values[feature] = st.slider(
                        label="Nilai",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(default_value),
                        step=0.1,
                        key=f"slider_{current_category}_{feature}",
                        label_visibility="collapsed"
                    )
            
            st.markdown("---")
            col_back, col_next = st.columns([1, 1])
            
            with col_back:
                back_button = st.form_submit_button("⬅️ Kembali", disabled=(st.session_state.current_step == 0))
            
            with col_next:
                if st.session_state.current_step == total_steps - 1:
                    next_button = st.form_submit_button("📋 Review Data")
                else:
                    next_button = st.form_submit_button("Selanjutnya ➡️")
            
            if back_button:
                st.session_state.current_step -= 1
                st.rerun()
            
            if next_button:
                # Simpan data
                save_form_data(current_category, form_values)
                
                if st.session_state.current_step == total_steps - 1:
                    # Tampilkan review
                    st.session_state.show_review = True
                else:
                    st.session_state.current_step += 1
                st.rerun()
    
    # Halaman Review Data
    elif st.session_state.show_review and not st.session_state.prediction_made:
        st.markdown("<h2 class='category-header'>📋 Review Data Input</h2>", unsafe_allow_html=True)
        st.info("📝 Periksa kembali data yang telah Anda masukkan sebelum melakukan prediksi.")
        
        # Tampilkan semua data yang telah diinput
        for category, features in CATEGORIES.items():
            st.markdown(f"### {category}")
            if category in st.session_state.form_data:
                category_data = st.session_state.form_data[category]
                
                # Buat dataframe untuk tampilan yang rapi
                df_display = pd.DataFrame([
                    {"Fitur": feat, "Nilai": category_data.get(feat, 0)} 
                    for feat in features
                ])
                st.dataframe(df_display, use_container_width=True, hide_index=True)
            st.markdown("---")
        
        # Tombol navigasi
        col_back, col_predict = st.columns([1, 1])
        
        with col_back:
            if st.button("⬅️ Kembali ke Input Terakhir", use_container_width=True):
                st.session_state.show_review = False
                st.rerun()
        
        with col_predict:
            if st.button("🔍 Prediksi Banjir", use_container_width=True, type="primary"):
                st.session_state.prediction_made = True
                st.rerun()
    
    # Tampilkan hasil prediksi
    elif st.session_state.prediction_made:
        st.markdown("<h2 class='category-header'>📊 Hasil Prediksi</h2>", unsafe_allow_html=True)
        
        # Load model
        model = load_model()
        
        if model is not None:
            # Gabungkan semua data
            all_data = {}
            for category in CATEGORIES.keys():
                if category in st.session_state.form_data:
                    all_data.update(st.session_state.form_data[category])
            
            # Urutan fitur sesuai dengan saat training model
            feature_order = [
                "MonsoonIntensity",
                "TopographyDrainage",
                "RiverManagement",
                "Deforestation",
                "Urbanization",
                "ClimateChange",
                "DamsQuality",
                "Siltation",
                "AgriculturalPractices",
                "Encroachments",
                "IneffectiveDisasterPreparedness",
                "DrainageSystems",
                "CoastalVulnerability",
                "Landslides",
                "Watersheds",
                "DeterioratingInfrastructure",
                "PopulationScore",
                "WetlandLoss",
                "InadequatePlanning",
                "PoliticalFactors"
            ]
            
            df_input = pd.DataFrame([all_data])[feature_order]
            
            # Prediksi
            try:
                prediction = model.predict(df_input)[0]
                risk_level, emoji, color = get_risk_level(prediction)
                
                # Tampilkan hasil
                st.markdown(f"""
                <div class='prediction-box'>
                    <h2>Probabilitas Banjir</h2>
                    <h1 style='color: {color}; font-size: 4rem;'>{emoji} {prediction:.2%}</h1>
                    <h2 style='color: {color};'>Risiko: {risk_level}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Rekomendasi
                st.markdown("### 💡 Rekomendasi")
                if prediction < 0.35:
                    st.success("✅ Risiko banjir rendah. Tetap pantau kondisi cuaca dan lingkungan.")
                elif prediction < 0.50:
                    st.warning("⚠️ Risiko banjir sedang. Persiapkan rencana evakuasi dan monitor perkembangan.")
                elif prediction < 0.65:
                    st.warning("⚠️ Risiko banjir tinggi! Segera lakukan mitigasi dan siapkan evakuasi.")
                else:
                    st.error("🚨 RISIKO BANJIR SANGAT TINGGI! Evakuasi segera dan hubungi pihak berwenang!")
                
                # Tampilkan detail input
                with st.expander("📋 Lihat Detail Input"):
                    for category, features in CATEGORIES.items():
                        st.markdown(f"**{category}**")
                        category_data = {feat: all_data[feat] for feat in features}
                        st.dataframe(pd.DataFrame([category_data]), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error saat melakukan prediksi: {str(e)}")
        
        # Tombol reset
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Prediksi Baru", use_container_width=True):
                st.session_state.current_step = 0
                st.session_state.form_data = {}
                st.session_state.prediction_made = False
                st.session_state.show_review = False
                st.rerun()

if __name__ == "__main__":
    main()