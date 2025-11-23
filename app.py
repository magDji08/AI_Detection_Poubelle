# %%writefile app.py
import os
import io
import tempfile
import shutil
import time
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
import cv2

# --- 1. Imports conditionnels et initialisation ---
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    st.error("La librairie 'ultralytics' est requise. Exécutez : pip install ultralytics opencv-python-headless")
    ULTRALYTICS_AVAILABLE = False
    class YOLO:
        def __init__(self, *args, **kwargs): pass
        def predict(self, *args, **kwargs): return []

MODEL_LOCAL = Path("best.pt")

# Configuration de la page Streamlit
st.set_page_config(
    page_title="🗑️ Détecteur de Poubelles",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS personnalisé optimisé pour la lisibilité ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    /* === VARIABLES DE COULEURS === */
    :root {
        --primary-color: #2E7D32;
        --secondary-color: #1B5E20;
        --accent-color: #4CAF50;
        --danger-color: #D32F2F;
        --warning-color: #F57C00;
        --success-color: #388E3C;
        --bg-light: #F5F7FA;
        --bg-white: #FFFFFF;
        --text-dark: #1A202C;
        --text-medium: #4A5568;
        --text-light: #718096;
        --border-color: #E2E8F0;
    }

    /* === BASE === */
    html, body, [class*="st-"] {
        font-family: 'Poppins', sans-serif;
        color: var(--text-dark);
    }

    .stApp {
        background: linear-gradient(135deg, #F5F7FA 0%, #E8EEF2 100%);
    }

    /* === TITRE PRINCIPAL === */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #2E7D32 0%, #4CAF50 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1.5rem;
        padding: 1.5rem;
        animation: slideDown 0.6s ease-out;
    }

    /* === CARTES === */
    .card {
        background: var(--bg-white);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
        transition: all 0.4s ease;
    }

    .card:hover {
        box-shadow: 0 8px 30px rgba(46, 125, 50, 0.15);
        transform: translateY(-4px);
    }

    .card h3 {
        color: var(--primary-color);
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .card p {
        color: var(--text-medium);
        line-height: 1.8;
        margin-bottom: 0.5rem;
        font-size: 1.05rem;
    }

    /* === SOUS-TITRES === */
    .sub-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--secondary-color);
        margin: 2.5rem 0 1.5rem 0;
        padding-left: 1.2rem;
        border-left: 5px solid var(--accent-color);
        animation: slideRight 0.5s ease-out;
    }

    /* === BADGES DE STATUT === */
    .status-badge {
        padding: 1rem 2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.2rem;
        text-align: center;
        display: inline-block;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        animation: fadeInScale 0.6s ease-out;
        letter-spacing: 0.5px;
    }

    .status-full {
        background: linear-gradient(135deg, #D32F2F 0%, #B71C1C 100%);
        color: white;
        animation: pulse 2s infinite;
    }

    .status-empty {
        background: linear-gradient(135deg, #388E3C 0%, #2E7D32 100%);
        color: white;
    }

    .status-unknown {
        background: linear-gradient(135deg, #F57C00 0%, #EF6C00 100%);
        color: white;
    }

    /* === SECTION UPLOAD AMÉLIORÉE === */
    .upload-box {
        background: white;
        border: 3px dashed var(--accent-color);
        border-radius: 20px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 2rem 0;
        transition: all 0.4s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.06);
    }

    .upload-box:hover {
        border-color: var(--primary-color);
        background: rgba(76, 175, 80, 0.03);
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(46, 125, 50, 0.15);
    }

    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        animation: bounce 2s infinite;
    }

    .upload-text {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--primary-color);
        margin-bottom: 0.5rem;
    }

    .upload-hint {
        font-size: 1rem;
        color: var(--text-medium);
        font-weight: 500;
    }

    /* CORRECTION CRITIQUE: Rendre TOUS les textes des uploaders VISIBLES */
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] div {
        color: var(--text-dark) !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }

    [data-testid="stFileUploader"] small {
        color: var(--text-medium) !important;
        font-size: 0.95rem !important;
    }

    /* Zone de drop */
    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 16px;
        padding: 2rem !important;
    }

    /* Bouton Browse files visible */
    [data-testid="stFileUploader"] button {
        background: var(--accent-color) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        padding: 0.7rem 1.5rem !important;
    }

    /* === CAMERA INPUT === */
    [data-testid="stCameraInput"] *,
    [data-testid="stCameraInput"] label,
    [data-testid="stCameraInput"] span,
    [data-testid="stCameraInput"] button {
        color: var(--text-dark) !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
    }

    [data-testid="stCameraInput"] button {
        background: var(--accent-color) !important;
        color: white !important;
        padding: 1rem 2rem !important;
        border-radius: 12px !important;
        font-size: 1.2rem !important;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
        transition: all 0.3s ease;
    }

    [data-testid="stCameraInput"] button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.4);
    }

    /* === WEBCAM SECTION === */
    .webcam-container {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 30px rgba(46, 125, 50, 0.15);
        animation: fadeIn 0.8s ease-out;
        border: 3px solid var(--accent-color);
        margin: 2rem 0;
    }

    .webcam-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-color);
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* === VIDEO SECTION === */
    .video-container {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 30px rgba(245, 124, 0, 0.15);
        animation: fadeIn 0.8s ease-out;
        border: 3px solid var(--warning-color);
        margin: 2rem 0;
    }

    .video-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--warning-color);
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    /* === SIDEBAR ORGANISÉE === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B5E20 0%, #2E7D32 100%);
        padding: 1.5rem 1rem;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Titre sidebar */
    [data-testid="stSidebar"] h1 {
        font-size: 1.8rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 2rem !important;
        padding-bottom: 1rem;
        border-bottom: 3px solid rgba(255,255,255,0.3);
    }

    /* Sections sidebar */
    [data-testid="stSidebar"] h2 {
        font-size: 1.3rem !important;
        font-weight: 700 !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        padding: 0.8rem 1rem;
        background: rgba(255,255,255,0.15);
        border-radius: 10px;
        border-left: 4px solid white;
    }

    [data-testid="stSidebar"] h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
    }

    /* Labels sidebar */
    [data-testid="stSidebar"] label {
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* Info box dans sidebar */
    .model-info {
        background: rgba(255, 255, 255, 0.2);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        border: 2px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
    }

    .sidebar-divider {
        height: 2px;
        background: rgba(255,255,255,0.2);
        margin: 1.5rem 0;
        border-radius: 2px;
    }

    /* === BOUTONS === */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-color) 0%, var(--primary-color) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.9rem 2rem;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(46, 125, 50, 0.3);
        letter-spacing: 0.5px;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(46, 125, 50, 0.5);
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--accent-color) 100%);
    }

    /* Bouton de téléchargement */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%) !important;
        color: white !important;
        font-weight: 700 !important;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%) !important;
        transform: translateY(-3px);
    }

    /* === BARRE DE PROGRESSION === */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent-color), var(--primary-color));
        height: 12px !important;
        border-radius: 10px;
    }

    /* === ANIMATIONS === */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }

    @keyframes fadeInScale {
        from { 
            opacity: 0; 
            transform: scale(0.9);
        }
        to { 
            opacity: 1; 
            transform: scale(1);
        }
    }

    @keyframes slideDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideRight {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.08); }
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }

    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: white;
        padding: 0.8rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.15rem;
        padding: 1rem 2.5rem;
        color: var(--text-medium) !important;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(46, 125, 50, 0.1);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--accent-color), var(--primary-color)) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
    }

    /* === EXPANDER === */
    .streamlit-expanderHeader {
        background: var(--bg-light) !important;
        border-radius: 10px;
        font-weight: 700;
        color: var(--primary-color) !important;
        font-size: 1.1rem;
    }

    /* === DATAFRAME === */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }

    /* === MESSAGES INFO === */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid;
        font-weight: 500;
        font-size: 1.05rem;
    }

</style>
""", unsafe_allow_html=True)

# --- Initialisation de l'état de session ---
if 'model_path' not in st.session_state:
    st.session_state.model_path = None
if 'model' not in st.session_state:
    st.session_state.model = None

# --- 2. Fonctions d'aide ---

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Charge le modèle YOLO et le met en cache."""
    if not ULTRALYTICS_AVAILABLE:
        return None
    try:
        with st.spinner("🔄 Chargement du modèle IA..."):
            model = YOLO(model_path)
            time.sleep(1)
        return model
    except Exception as e:
        st.error(f"Échec du chargement du modèle : {e}")
        return None

def run_image_inference(model, img_path, conf=0.25, iou=0.45):
    """Effectue l'inférence sur l'image et analyse le résultat."""
    try:
        results = model.predict(source=str(img_path), conf=conf, iou=iou, save=False)

        if not results:
            st.warning("Aucune prédiction retournée par le modèle.")
            return None, None, "Aucun résultat", "status-unknown"

        r = results[0]
        annotated = r.plot()

        boxes = []
        try:
            xyxy = r.boxes.xyxy.tolist()
            confs = r.boxes.conf.tolist()
            classes = r.boxes.cls.tolist()
            names = r.names if hasattr(r, 'names') else (model.names if hasattr(model, 'names') else {})

            for b, c, cl in zip(xyxy, confs, classes):
                boxes.append({
                    "xmin": int(b[0]), "ymin": int(b[1]), "xmax": int(b[2]), "ymax": int(b[3]),
                    "conf": float(c), "class_id": int(cl), "label": names.get(int(cl), str(cl))
                })
        except Exception:
            boxes = []

        status = "Inconnu"
        status_class = "status-unknown"

        if boxes:
            classes_detected = [box['label'].lower() for box in boxes]

            if any(k in c for c in classes_detected for k in ['plein', 'full', 'filled']):
                status = '🔴 Pleine'
                status_class = "status-full"
            elif any(k in c for c in classes_detected for k in ['vide', 'empty', 'empty-bin']):
                status = '🟢 Vide'
                status_class = "status-empty"
            else:
                status = '🟡 Poubelle détectée (état incertain)'
                status_class = "status-unknown"
        else:
            status = '⚪ Aucune poubelle détectée'
            status_class = "status-unknown"

        return annotated, boxes, status, status_class

    except Exception as e:
        st.error(f"Erreur d'inférence : {e}")
        return None, None, 'Erreur', "status-unknown"

def extract_frames_from_video(video_path, max_frames=6):
    """Extrait des frames d'une vidéo pour prévisualisation."""
    frames = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Impossible d'ouvrir le fichier vidéo.")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, total_frames // max_frames)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0 and len(frames) < max_frames:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            frame_count += 1
            if len(frames) >= max_frames:
                break

        cap.release()
        return frames
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des frames: {e}")
        return []

def process_video_with_progress(model, video_path, conf=0.25, iou=0.45):
    """Traite la vidéo avec une barre de progression."""

    progress_container = st.container()

    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()

    run_name = f"video_run_{time.time_ns()}"

    try:
        status_text.info("🔄 Démarrage du traitement vidéo...")
        progress_bar.progress(10)

        status_text.info("🔄 Analyse des frames avec l'IA...")
        progress_bar.progress(40)

        model.predict(source=str(video_path), conf=conf, iou=iou, save=True, name=run_name, exist_ok=True)

        status_text.info("🔄 Finalisation et encodage...")
        progress_bar.progress(80)

        output_dir = Path(f'runs/detect/{run_name}')
        video_out = next((p for p in output_dir.glob('*') if p.suffix.lower() in ['.mp4', '.mov', '.avi']), None)

        progress_bar.progress(100)
        progress_container.empty()

        if video_out and video_out.exists():
            return str(video_out)
        else:
            st.warning("Vidéo traitée, mais le fichier de sortie n'a pas été trouvé.")
            return None

    except Exception as e:
        st.error(f"Échec du traitement vidéo : {e}")
        progress_container.empty()
        return None

def cleanup_temp_file(path):
    """Supprime un fichier temporaire."""
    if path and Path(path).exists():
        try:
            os.unlink(path)
        except Exception:
            pass

# --- 3. Application Principale ---

def main():
    if not ULTRALYTICS_AVAILABLE:
        st.stop()

    st.markdown('<h1 class="main-header">🗑️ Détecteur Intelligent de Poubelles</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🎯 Bienvenue dans l'analyseur IA de poubelles</h3>
        <p>Cette application utilise un modèle <b>YOLO</b> personnalisé pour détecter et classifier automatiquement l'état de remplissage des poubelles.</p>
        <p>✨ <b>Fonctionnalités :</b> Détection en temps réel, analyse d'images et vidéos, classification Plein/Vide</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Sidebar Organisée ---
    with st.sidebar:
        st.markdown("# ⚙️ Configuration")
        
        # SECTION 1: Modèle
        st.markdown("## 📦 Modèle")
        st.markdown('<div class="model-info">✅ Modèle actif : <b>best.pt</b></div>', unsafe_allow_html=True)
        
        # Télécharger le modèle
        if MODEL_LOCAL.exists():
            with open(MODEL_LOCAL, 'rb') as f:
                st.download_button(
                    "💾 Télécharger best.pt",
                    data=f,
                    file_name="best.pt",
                    mime="application/octet-stream",
                    use_container_width=True,
                    key="download_model"
                )
        
        # Uploader un nouveau modèle (optionnel)
        with st.expander("🔄 Charger un autre modèle", expanded=False):
            uploaded_model = st.file_uploader("Fichier .pt", type=["pt"], label_visibility="collapsed", key="upload_model")
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        # SECTION 2: Paramètres de détection
        st.markdown("## 🎚️ Paramètres")
        conf_thres = st.slider(
            "🎯 Confiance", 
            0.0, 1.0, 0.50, 0.05,
            help="Niveau de confiance minimum",
            key="conf_slider"
        )
        iou_thres = st.slider(
            "🔲 NMS (IoU)", 
            0.0, 1.0, 0.50, 0.05,
            help="Suppression des boîtes superposées",
            key="iou_slider"
        )
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        # SECTION 3: Actions
        st.markdown("## 🔧 Actions")
        if st.button("🔄 Recharger le modèle", use_container_width=True, key="reload_btn"):
            load_model.clear()
            st.session_state.model_path = None
            st.session_state.model = None
            st.rerun()

    # --- Chargement du Modèle ---
    new_model_path = None

    if 'uploaded_model' in locals() and uploaded_model is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as t:
            t.write(uploaded_model.read())
            new_model_path = t.name
        st.sidebar.success("✅ Modèle personnalisé chargé")
    elif MODEL_LOCAL.exists():
        new_model_path = str(MODEL_LOCAL)
    else:
        st.sidebar.error("❌ Fichier best.pt introuvable")

    if new_model_path and st.session_state.model_path != new_model_path:
        if st.session_state.model_path and "tmp" in st.session_state.model_path:
             cleanup_temp_file(st.session_state.model_path)

        st.session_state.model = load_model(new_model_path)
        st.session_state.model_path = new_model_path

    model = st.session_state.model

    if model is None:
        st.sidebar.error("❌ Modèle non chargé")
    else:
        try:
            names = getattr(model, 'names', None)
            if names:
                st.sidebar.success(f"✅ Classes : {', '.join(names.values())}")
            else:
                st.sidebar.success("✅ Modèle opérationnel")
        except Exception:
            st.sidebar.success("✅ Modèle chargé")

    # --- Analyse avec 3 onglets: Fichier, Webcam, Vidéo ---
    st.markdown('<div class="sub-header">📸 Analyse</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📁 Fichier", "📷 Webcam", "🎥 Vidéo"])

    # ============ ONGLET 1: FICHIER ============
    with tab1:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div class="upload-box">
                <div class="upload-icon">📤</div>
                <div class="upload-text">Glissez-déposez votre image</div>
                <div class="upload-hint">ou cliquez pour parcourir</div>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_image = st.file_uploader(
                "Sélectionnez une image", 
                type=["jpg", "jpeg", "png"], 
                key="image_uploader",
                help="Formats : JPG, JPEG, PNG (max 200MB)"
            )

            if uploaded_image and model:
                image = Image.open(uploaded_image)
                st.image(image, caption='📷 Image originale', use_container_width=True)

        with col2:
            if uploaded_image and model:
                tmp_img_path = None
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as t:
                    image.save(t, format='JPEG')
                    tmp_img_path = t.name

                with st.spinner("🔍 Analyse IA en cours..."):
                    annotated, boxes, status, status_class = run_image_inference(model, tmp_img_path, conf=conf_thres, iou=iou_thres)

                cleanup_temp_file(tmp_img_path)

                if annotated is not None:
                    st.markdown(f'<div class="status-badge {status_class}">{status}</div>', unsafe_allow_html=True)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.image(annotated, caption='✨ Résultat de l\'analyse', use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    if boxes:
                        with st.expander("📊 Détails des détections", expanded=True):
                            df_boxes = [{
                                "🏷️ Classe": box['label'],
                                "✅ Confiance": f"{box['conf']:.2%}",
                                "📍 Position": f"({box['xmin']}, {box['ymin']})"
                            } for box in boxes]
                            st.dataframe(df_boxes, hide_index=True, use_container_width=True)

                    buf = io.BytesIO()
                    Image.fromarray(annotated.astype('uint8')).save(buf, format='JPEG')
                    buf.seek(0)
                    st.download_button("💾 Télécharger le résultat", data=buf, file_name="analyse_poubelle.jpg", mime='image/jpeg', use_container_width=True, key="dl_file")
                else:
                    st.warning("⚠️ Aucune détection au-dessus du seuil de confiance")

            elif uploaded_image and not model:
                st.error("❌ Veuillez charger un modèle")
            
            elif not uploaded_image:
                st.info("👈 Téléversez une image pour commencer")

    # ============ ONGLET 2: WEBCAM ============
    with tab2:
        st.markdown('<div class="webcam-container">', unsafe_allow_html=True)
        st.markdown('<div class="webcam-title">📷 Capture en temps réel</div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #2E7D32; font-size: 1.1rem; font-weight: 600; margin-bottom: 2rem;">Capturez une image instantanée pour une analyse immédiate</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 15px; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">📸</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: #2E7D32; margin-bottom: 0.5rem;">Prendre une photo</div>
                <div style="font-size: 1rem; color: #4A5568;">Cliquez sur le bouton ci-dessous</div>
            </div>
            """, unsafe_allow_html=True)
            
            camera_image = st.camera_input("📸 Prendre une photo", key="camera", label_visibility="visible")

        with col2:
            if camera_image and model:
                image = Image.open(camera_image)

                tmp_img_path = None
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as t:
                    image.save(t, format='JPEG')
                    tmp_img_path = t.name

                with st.spinner("🔍 Analyse IA en cours..."):
                    annotated, boxes, status, status_class = run_image_inference(model, tmp_img_path, conf=conf_thres, iou=iou_thres)

                cleanup_temp_file(tmp_img_path)

                if annotated is not None:
                    st.markdown(f'<div class="status-badge {status_class}">{status}</div>', unsafe_allow_html=True)

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.image(annotated, caption='✨ Résultat de l\'analyse', use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                    if boxes:
                        with st.expander("📊 Détails des détections", expanded=True):
                            df_boxes = [{
                                "🏷️ Classe": box['label'],
                                "✅ Confiance": f"{box['conf']:.2%}",
                                "📍 Position": f"({box['xmin']}, {box['ymin']})"
                            } for box in boxes]
                            st.dataframe(df_boxes, hide_index=True, use_container_width=True)

                    buf = io.BytesIO()
                    Image.fromarray(annotated.astype('uint8')).save(buf, format='JPEG')
                    buf.seek(0)
                    st.download_button("💾 Télécharger le résultat", data=buf, file_name="analyse_webcam.jpg", mime='image/jpeg', use_container_width=True, key="dl_webcam")
                else:
                    st.warning("⚠️ Aucune détection au-dessus du seuil de confiance")

            elif camera_image and not model:
                st.error("❌ Veuillez charger un modèle")
            
            elif not camera_image:
                st.info("👆 Capturez une image avec votre webcam pour commencer l'analyse")

    # ============ ONGLET 3: VIDÉO ============
    with tab3:
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        st.markdown('<div class="video-title">🎬 Analyseur Vidéo Intelligent</div>', unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #E65100; font-size: 1.1rem; font-weight: 600; margin-bottom: 2rem;">Téléversez votre vidéo pour une analyse frame par frame</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="upload-box">
            <div class="upload-icon">🎥</div>
            <div class="upload-text">Glissez-déposez votre vidéo</div>
            <div class="upload-hint">ou cliquez pour parcourir</div>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_video = st.file_uploader(
            "Sélectionnez une vidéo", 
            type=["mp4", "mov", "avi"], 
            key="video_uploader",
            help="Formats : MP4, MOV, AVI (max 200MB)"
        )

        if uploaded_video and model:
            col_v1, col_v2 = st.columns([1, 1])
            
            with col_v1:
                st.markdown("### 📹 Vidéo Originale")
                st.video(uploaded_video)
            
            with col_v2:
                st.markdown("### ⚙️ Options")
                show_video_frames = st.checkbox("🎞 Aperçu des frames", value=True, help="Prévisualisation de 6 frames extraites", key="frames_check")
                
                if show_video_frames:
                    with st.spinner("🔄 Extraction des frames..."):
                        tmp_vid_path = None
                        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as t:
                            t.write(uploaded_video.read())
                            tmp_vid_path = t.name

                        frames = extract_frames_from_video(tmp_vid_path, max_frames=6)
                        cleanup_temp_file(tmp_vid_path)

                        if frames:
                            st.markdown("**🎞 Aperçu :**")
                            cols = st.columns(3)
                            for i, frame in enumerate(frames):
                                with cols[i % 3]:
                                    st.image(frame, caption=f"Frame {i+1}", use_container_width=True)

            st.markdown("---")
            
            # Centrer le bouton d'analyse
            col_center1, col_center2, col_center3 = st.columns([1, 2, 1])
            with col_center2:
                analyze_btn = st.button("🚀 Lancer l'analyse vidéo complète", use_container_width=True, key="video_btn")
            
            if analyze_btn:
                tmp_vid_path = None
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_video.name).suffix) as t:
                    t.write(uploaded_video.getvalue())
                    tmp_vid_path = t.name

                output_video_path = process_video_with_progress(model, tmp_vid_path, conf=conf_thres, iou=iou_thres)
                cleanup_temp_file(tmp_vid_path)

                if output_video_path:
                    st.balloons()
                    
                    # Scroll automatique vers les résultats
                    st.markdown('<div id="results"></div>', unsafe_allow_html=True)
                    
                    st.success("✅ Analyse terminée avec succès !")
                    
                    col_result1, col_result2 = st.columns([1, 1])
                    
                    with col_result1:
                        st.markdown("### 📹 Vidéo Analysée")
                        st.video(output_video_path)
                    
                    with col_result2:
                        st.markdown("### 💾 Téléchargement")
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%); padding: 2rem; border-radius: 15px; margin-top: 1rem; border: 2px solid #4CAF50;">
                            <p style="color: #2E7D32; font-weight: 700; font-size: 1.2rem; margin-bottom: 1rem; text-align: center;">
                            ✨ Votre vidéo est prête !
                            </p>
                            <p style="color: #4A5568; font-size: 1rem; text-align: center;">
                            Téléchargez-la pour la conserver ou la partager avec votre équipe.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with open(output_video_path, 'rb') as f:
                            st.download_button(
                                '💾 Télécharger la vidéo analysée',
                                data=f,
                                file_name=f"video_analysee_{int(time.time())}.mp4",
                                mime='video/mp4',
                                use_container_width=True,
                                key="dl_video"
                            )
                    
                    # Script pour scroller vers les résultats
                    st.markdown("""
                    <script>
                        const element = document.getElementById('results');
                        if (element) {
                            element.scrollIntoView({behavior: 'smooth', block: 'center'});
                        }
                    </script>
                    """, unsafe_allow_html=True)

        elif uploaded_video and not model:
            st.error("❌ Veuillez charger un modèle avant de lancer l'analyse")
        
        elif not uploaded_video:
            st.info("👆 Téléversez une vidéo pour commencer l'analyse")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Footer ---
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; padding: 2.5rem 0;">
        <p style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">🗑️ Détecteur Intelligent de Poubelles</p>
        <p style="font-size: 0.95rem; color: #4A5568;">Propulsé par <b style="color: #2E7D32;">YOLO</b> et <b style="color: #2E7D32;">Streamlit</b></p>
        <p style="font-size: 0.9rem; margin-top: 1rem; color: #718096;">Pour une gestion optimale des déchets 🌍♻️</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()