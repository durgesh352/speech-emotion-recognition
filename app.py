"""
Bilingual Speech Emotion Recognition (SER) Interactive Dashboard
Built with Streamlit, Plotly, Librosa, and Scikit-Learn.
Supports Hindi and English, Emotion Intensity, Confidence Scores,
Dynamic Emotion Timeline, and Multimodal NLP Fusion.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

# Configure page settings
st.set_page_config(
    page_title="Speech Emotion Recognition | Hindi & English",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.config import (
    EMOTIONS,
    EMOTION_META,
    LANGUAGES,
    SAMPLE_AUDIO_DIR,
    MODEL_FILE
)
from src.models.ser_pipeline import SpeechEmotionRecognizer
from src.models.model_trainer import SpeechEmotionTrainer
from src.utils.audio_generator import generate_sample_audio_files
from src.utils.visualizer import (
    plot_confidence_distribution,
    plot_emotion_timeline,
    plot_emotion_radar,
    plot_intensity_gauge,
    plot_waveform
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366F1, #EC4899, #F59E0B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Initializes or trains the emotion recognizer."""
    recognizer = SpeechEmotionRecognizer()
    if not recognizer.is_loaded:
        with st.spinner("Initializing bilingual model for first-time use... (generating assets)"):
            trainer = SpeechEmotionTrainer()
            X, y = trainer.generate_bilingual_training_dataset(samples_per_emotion=50)
            trainer.train(X, y)
            generate_sample_audio_files()
            recognizer = SpeechEmotionRecognizer()
    return recognizer


def main():
    # Header
    st.markdown('<div class="main-title">🎙️ Bilingual Speech Emotion Recognition System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Multimodal Audio Prosody + NLP Emotion Analysis | 🇮🇳 Hindi & 🇬🇧 English Supported</div>', unsafe_allow_html=True)

    # Initialize Recognizer
    recognizer = load_pipeline()

    # Sidebar
    st.sidebar.header("⚙️ Configuration & Input")
    
    # Language Selection
    lang_choice = st.sidebar.selectbox(
        "🌐 Target Language",
        options=["auto", "en", "hi"],
        format_func=lambda x: f"{'🌐' if x=='auto' else '🇬🇧' if x=='en' else '🇮🇳'} {LANGUAGES[x]}"
    )

    # Input Method Selection
    input_mode = st.sidebar.radio(
        "📥 Audio Input Source",
        options=["Preset Samples (Demo)", "Upload Audio File", "Live Microphone Recording"]
    )

    selected_audio = None
    audio_bytes = None
    default_transcript = ""

    # Ensure sample audio files exist
    if not SAMPLE_AUDIO_DIR.exists():
        generate_sample_audio_files()

    if input_mode == "Preset Samples (Demo)":
        sample_lang = st.sidebar.selectbox("Sample Language", ["English Samples", "Hindi Samples (हिंदी)"])
        lang_folder = "english" if "English" in sample_lang else "hindi"
        folder_path = SAMPLE_AUDIO_DIR / lang_folder
        
        sample_files = list(folder_path.glob("*.wav"))
        if not sample_files:
            generate_sample_audio_files()
            sample_files = list(folder_path.glob("*.wav"))

        options_dict = {f.name.replace(".wav", "").replace("_sample", "").title(): f for f in sample_files}
        chosen_sample_name = st.sidebar.selectbox("Choose Emotion Sample", list(options_dict.keys()))
        selected_audio = options_dict[chosen_sample_name]

        # Suggest transcript based on sample
        if "Hindi" in sample_lang:
            if "Khushi" in chosen_sample_name:
                default_transcript = "आज मैं बहुत खुश हूँ, हमने प्रतियोगिता जीत ली!"
            elif "Gussa" in chosen_sample_name:
                default_transcript = "मुझे तुमसे बहुत गुस्सा आ रहा है, बकवास बंद करो!"
            elif "Udasi" in chosen_sample_name:
                default_transcript = "मुझे बहुत अकेलापन और उदासी महसूस हो रही है।"
            elif "Darr" in chosen_sample_name:
                default_transcript = "वहां बहुत अंधेरा और खतरा है, मुझे डर लग रहा है!"
            elif "Shant" in chosen_sample_name:
                default_transcript = "सब कुछ ठीक है, सामान्य बातचीत चल रही है।"
            elif "Aashcharya" in chosen_sample_name:
                default_transcript = "क्या! सच में? यह तो अद्भुत और अविश्वसनीय है!"
            else:
                default_transcript = "यह बहुत घिनौना और खराब है।"
        else:
            if "Happy" in chosen_sample_name:
                default_transcript = "I am absolutely thrilled and overjoyed with this wonderful news!"
            elif "Angry" in chosen_sample_name:
                default_transcript = "I am so furious right now, this is completely unacceptable nonsense!"
            elif "Sad" in chosen_sample_name:
                default_transcript = "I feel so lonely, brokenhearted and sad today."
            elif "Fear" in chosen_sample_name:
                default_transcript = "I am terrified, please help me there is danger everywhere!"
            elif "Neutral" in chosen_sample_name:
                default_transcript = "This is a regular neutral update regarding the scheduled meeting."
            elif "Surprise" in chosen_sample_name:
                default_transcript = "Wow! Really? That is completely unexpected and unbelievable!"
            else:
                default_transcript = "That is so disgusting, nasty and revolting."

    elif input_mode == "Upload Audio File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload an audio clip",
            type=["wav", "mp3", "ogg", "flac"]
        )
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            selected_audio = audio_bytes

    else:  # Live Microphone
        st.sidebar.info("Click 'Record' below to capture speech directly from your microphone.")
        mic_audio = st.sidebar.audio_input("Record Speech")
        if mic_audio is not None:
            audio_bytes = mic_audio.read()
            selected_audio = audio_bytes

    # NLP Semantic Transcript Fusion options
    st.sidebar.markdown("---")
    st.sidebar.subheader("💬 NLP Multimodal Fusion")
    enable_nlp = st.sidebar.checkbox("Enable Spoken Text / NLP Fusion", value=True)
    user_transcript = ""
    audio_weight = 0.70

    if enable_nlp:
        user_transcript = st.sidebar.text_area(
            "Spoken Transcript (Hindi / English / Hinglish)",
            value=default_transcript,
            help="Enter what was said in the audio to fuse acoustic prosody with semantic emotion analysis."
        )
        audio_weight = st.sidebar.slider(
            "Audio vs NLP Weight Balance",
            min_value=0.1,
            max_value=1.0,
            value=0.70,
            step=0.05,
            format="%d%% Audio",
            help="70% Audio + 30% NLP is recommended for speech emotion analysis."
        )

    # Main Area Processing
    if selected_audio is None:
        st.info("👈 Please select a sample audio, upload a file, or record your voice from the sidebar to start.")
        return

    # Run Prediction
    with st.spinner("Analyzing acoustic features, emotion intensity, and timeline..."):
        try:
            results = recognizer.predict_audio(
                audio_input=selected_audio,
                language=lang_choice,
                transcript=user_transcript if enable_nlp else None,
                audio_weight=audio_weight
            )
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

    pred_emo = results["predicted_emotion"]
    meta = EMOTION_META.get(pred_emo, {})
    color = meta.get("color", "#6366F1")
    emoji = meta.get("emoji", "🎙️")
    hi_name = meta.get("hi_name", pred_emo)
    intensity = results["emotion_intensity"]

    # Row 1: Audio Playback & KPI Cards
    col_audio, col_kpi1, col_kpi2, col_kpi3 = st.columns([1.3, 1.2, 1, 1])

    with col_audio:
        st.subheader("🔊 Audio Playback")
        if isinstance(selected_audio, (str, Path)):
            st.audio(str(selected_audio))
        elif audio_bytes:
            st.audio(audio_bytes)
        st.caption(f"⏱️ Duration: **{results['duration_seconds']}s** | Sample Rate: **{results['sample_rate']} Hz**")

    with col_kpi1:
        st.markdown(
            f"""
            <div class="metric-card" style="border-top: 4px solid {color};">
                <span style="font-size: 2.2rem;">{emoji}</span>
                <h4 style="margin: 0.2rem 0; color: {color};">{pred_emo}</h4>
                <div style="color: #CBD5E1; font-size: 0.95rem;">{hi_name}</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.3rem;">Primary Emotion</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_kpi2:
        conf_pct = results["confidence_percentage"]
        st.markdown(
            f"""
            <div class="metric-card" style="border-top: 4px solid #38BDF8;">
                <span style="font-size: 2.2rem;">🎯</span>
                <h3 style="margin: 0.2rem 0; color: #38BDF8;">{conf_pct}%</h3>
                <div style="color: #CBD5E1; font-size: 0.95rem;">Confidence</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.3rem;">Calibrated Posterior</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_kpi3:
        inten_pct = intensity["intensity_percentage"]
        inten_label = intensity["level_en"]
        inten_hi = intensity["level_hi"]
        st.markdown(
            f"""
            <div class="metric-card" style="border-top: 4px solid #F59E0B;">
                <span style="font-size: 2.2rem;">⚡</span>
                <h3 style="margin: 0.2rem 0; color: #F59E0B;">{inten_pct}%</h3>
                <div style="color: #CBD5E1; font-size: 0.95rem;">{inten_label} ({inten_hi})</div>
                <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 0.3rem;">Vocal Intensity</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Waveform and Intensity Dial
    col_wave, col_dial = st.columns([2, 1])
    with col_wave:
        st.plotly_chart(plot_waveform(results["audio_waveform"], results["sample_rate"]), use_container_width=True)
    with col_dial:
        st.plotly_chart(
            plot_intensity_gauge(intensity["intensity_percentage"], intensity["level_en"], intensity["level_hi"]),
            use_container_width=True
        )

    # Row 2: Prediction Charts & Emotion Radar
    st.subheader("📊 Emotion Confidence Distribution & Acoustic Footprint")
    tab_bar, tab_radar, tab_table = st.tabs(["📊 Confidence Bar Chart", "🕸️ Radar Emotion Fingerprint", "📋 Probability Table"])

    with tab_bar:
        st.plotly_chart(
            plot_confidence_distribution(results["emotion_probabilities"], pred_emo),
            use_container_width=True
        )

    with tab_radar:
        st.plotly_chart(
            plot_emotion_radar(results["emotion_probabilities"]),
            use_container_width=True
        )

    with tab_table:
        prob_df = pd.DataFrame([
            {
                "Emotion": f"{EMOTION_META[e]['emoji']} {e}",
                "Hindi Name": EMOTION_META[e]["hi_name"],
                "Confidence Score": f"{prob:.4f}",
                "Percentage": f"{prob*100:.1f}%"
            }
            for e, prob in sorted(results["emotion_probabilities"].items(), key=lambda x: x[1], reverse=True)
        ])
        st.dataframe(prob_df, use_container_width=True, hide_index=True)

    # Row 3: DYNAMIC EMOTION TIMELINE
    st.markdown("---")
    st.subheader("📈 Dynamic Emotion Timeline (Fluctuations Across Audio)")
    st.markdown("Tracks how the speaker's emotions and vocal intensity evolve second-by-second using sliding acoustic windows.")

    timeline_data = results["timeline"]
    if timeline_data:
        st.plotly_chart(plot_emotion_timeline(timeline_data), use_container_width=True)

        with st.expander("🔍 View Detailed Temporal Segments Data"):
            timeline_df = pd.DataFrame(timeline_data)
            display_cols = ["start_time", "end_time", "dominant_emotion", "confidence", "intensity_percentage", "intensity_level"]
            st.dataframe(timeline_df[display_cols], use_container_width=True)

    # Row 4: NLP Multimodal Insights (if enabled)
    if results.get("nlp_analysis") and results.get("multimodal_result"):
        st.markdown("---")
        st.subheader("💬 NLP Transcript & Multimodal Fusion Analysis")
        nlp_res = results["nlp_analysis"]
        multi_res = results["multimodal_result"]

        col_nlp1, col_nlp2 = st.columns(2)
        with col_nlp1:
            st.markdown(f"**Spoken Transcript:** *\"{user_transcript}\"*")
            st.markdown(f"- **Detected Language:** `{'Hindi (हिंदी)' if nlp_res['detected_language']=='hi' else 'English'}`")
            st.markdown(f"- **NLP Text Dominant Emotion:** `{nlp_res['primary_emotion']}` (Confidence: {nlp_res['confidence']*100:.1f}%)")
            keywords_str = ", ".join(nlp_res['detected_keywords']) if nlp_res['detected_keywords'] else "None"
            st.markdown(f"- **Emotional Keywords Detected:** `{keywords_str}`")

        with col_nlp2:
            st.markdown(f"- **Acoustic Top Prediction:** `{multi_res['acoustic_top']}`")
            st.markdown(f"- **NLP Semantic Top Prediction:** `{multi_res['nlp_top']}`")
            is_agree = multi_res['modal_agreement']
            st.markdown(
                f"- **Modal Congruence:** {'✅ **Tone & Words Agree**' if is_agree else '⚠️ **Tone & Words Contrast (Potential Irony/Sarcasm)**'}"
            )
            st.markdown(f"- **Fused Prediction:** **{multi_res['fused_emotion']}** ({multi_res['confidence_percentage']}%)")

    # Row 5: Retraining & Evaluation Studio
    st.markdown("---")
    with st.expander("🛠️ Model Studio & Retraining (Hindi + English)"):
        st.markdown("Retrain the bilingual model with synthetic or custom WAV dataset.")
        train_samples = st.slider("Samples per emotion for training", min_value=30, max_value=120, value=60)
        
        if st.button("🚀 Retrain Model Now"):
            with st.spinner("Training bilingual model across English and Hindi prosodic profiles..."):
                trainer = SpeechEmotionTrainer()
                X, y = trainer.generate_bilingual_training_dataset(samples_per_emotion=train_samples)
                meta_res = trainer.train(X, y)
                st.success(f"Model retrained successfully! Test Accuracy: {meta_res['accuracy']*100:.2f}%")
                st.cache_resource.clear()
                st.rerun()


if __name__ == "__main__":
    main()
