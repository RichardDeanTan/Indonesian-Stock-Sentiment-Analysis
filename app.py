import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import json

MODEL_NAME = "RichTan/IndoStockSentiment"
MAX_SEQUENCE_LENGTH = 64
MAX_BATCH_SIZE = 1000

# Sentiment labels
SENTIMENT_LABELS = ['Negative', 'Neutral', 'Positive']
SENTIMENT_COLORS = ['#ff4757', '#ffa502', '#2ed573']

st.set_page_config(
    page_title="Indonesian Stock Sentiment Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model_and_tokenizer(model_name):
    try:
        with st.spinner("Loading model and tokenizer from Hugging Face..."):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Set device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            model.eval()
            
            return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {str(e)}")
        return None, None

def preprocess_text(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'\[USERNAME\]|\[URL\]|\[HASHTAG\]', '', text, flags=re.IGNORECASE)
    return text.strip()

def predict_sentiment(text, model, tokenizer, max_length=64):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        processed_text = preprocess_text(text)
        inputs = tokenizer(
            processed_text,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probabilities, dim=-1).item()
            confidence_scores = probabilities.cpu().numpy()[0]
        
        return {
            'predicted_sentiment': SENTIMENT_LABELS[predicted_class],
            'confidence_scores': {label: float(score) for label, score in zip(SENTIMENT_LABELS, confidence_scores)},
            'processed_text': processed_text
        }
    except Exception as e:
        st.error(f"Error predicting sentiment: {str(e)}")
        return None

def predict_batch_sentiment(texts, model, tokenizer, max_length=64):
    results = []
    
    progress_bar = st.progress(0)
    for i, text in enumerate(texts):
        result = predict_sentiment(text, model, tokenizer, max_length)
        if result:
            results.append({
                'original_text': text,
                'processed_text': result['processed_text'],
                'predicted_sentiment': result['predicted_sentiment'],
                'confidence_negative': result['confidence_scores']['Negative'],
                'confidence_neutral': result['confidence_scores']['Neutral'],
                'confidence_positive': result['confidence_scores']['Positive']
            })
        progress_bar.progress((i + 1) / len(texts))
    
    return results

def create_confidence_chart(confidence_scores):
    labels = list(confidence_scores.keys())
    values = list(confidence_scores.values())
    colors = SENTIMENT_COLORS
    
    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{v:.3f}" for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Confidence Scores",
        xaxis_title="Sentiment",
        yaxis_title="Confidence Score",
        showlegend=False,
        height=300
    )
    
    return fig

def create_batch_distribution_chart(results_df):
    sentiment_counts = results_df['predicted_sentiment'].value_counts()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker_colors=[SENTIMENT_COLORS[SENTIMENT_LABELS.index(label)] for label in sentiment_counts.index]
        )
    ])
    
    fig.update_layout(
        title="Sentiment Distribution",
        height=400
    )
    
    return fig

def main():
    st.title("Indonesian Stock Sentiment Analyzer")
    st.markdown("""
    Aplikasi ini menggunakan fine-tuned **IndoBERTweet** model untuk menganalisis sentimen teks finansial Indonesia.
    Model dapat mengklasifikasikan teks menjadi 3 kategori: **Positive**, **Neutral**, atau **Negative**.
    """)
    
    # Load model
    with st.container():
        model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    
    if model is None or tokenizer is None:
        st.error("Failed to load model from Hugging Face")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Performance")
        
        st.info(f"""
        **🤗 Hugging Face Model:** `{MODEL_NAME}`
        - **Language:** Indonesian
        - **Domain:** Stock Market Finance
        - **Classes:** Positive, Neutral, Negative
        - **Metric Evaluation:** F1-Macro
        """)
        
        st.markdown("**Baseline Test Results:**")
        st.markdown("""
        - **TF-IDF SVM:** 73.89%
        - **IndoBERTweet:** 86.85% ⭐
        - **Llama 3.2 - 1B:** 80.22%
        """)

        st.markdown("**Fine-Tuned Test Results:**")
        st.success("**IndoBERTweet:** 87.53%")
        
        st.markdown("---")
        st.header("⚙️ Model Settings")
        
        max_length = st.slider(
            "Max Sequence Length",
            min_value=32,
            max_value=256,
            value=MAX_SEQUENCE_LENGTH,
            help="Maximum token length for model input"
        )
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Minimum confidence score to highlight predictions"
        )
        
        st.markdown("---")
        st.header("💡 Sample Texts")
        
        st.markdown("**Positive:**")
        st.code("[USERNAME] saham bca nya menyusul ya 🙂")
        
        st.markdown("**Neutral:**")
        st.code("[USERNAME] [USERNAME] pak buat sekuritas mirae apakah bisa pakai akun blu by bca?")
        
        st.markdown("**Negative:**")
        st.code("IHSG Dibuka Melemah Awal Pekan, Saham Bank Jumbo BMRI, BBCA, BBRI Berguguran [URL]")
    
    # === Main content ===
    tab1, tab2 = st.tabs(["📝 Single Text Analysis", "📁 Batch Analysis"])
    
    with tab1:
        st.header("📝 Single Text Analysis")
        
        input_option = st.radio(
            "Choose input method:",
            ["Type/Paste Text", "Use Sample"],
            horizontal=True
        )
        
        if input_option == "Use Sample":
            sample_option = st.selectbox(
                "Select sample:",
                ["Positive", "Neutral", "Negative"]
            )
            
            sample_texts = {
                "Positive": "[USERNAME] saham bca nya menyusul ya 🙂",
                "Neutral": "[USERNAME] [USERNAME] pak buat sekuritas mirae apakah bisa pakai akun blu by bca?",
                "Negative": "IHSG Dibuka Melemah Awal Pekan, Saham Bank Jumbo BMRI, BBCA, BBRI Berguguran [URL]"
            }
            
            input_text = st.text_area(
                "Indonesian Stock/Finance Text:",
                value=sample_texts[sample_option],
                height=100,
                help="Edit this sample text or replace with your own"
            )
        else:
            input_text = st.text_area(
                "Indonesian Stock/Finance Text:",
                height=100,
                placeholder="Masukkan teks bahasa Indonesia tentang saham/finansial di sini...",
                help="Paste your Indonesian stock/finance text here"
            )
        
        if input_text:
            word_count = len(input_text.split())
            st.caption(f"Words: {word_count}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            analyze_button = st.button(
                "🔍 Analyze Sentiment",
                type="primary",
                disabled=not input_text.strip(),
                use_container_width=True
            )
        
        if analyze_button and input_text.strip():
            with st.spinner("Analyzing sentiment..."):
                result = predict_sentiment(input_text, model, tokenizer, max_length)
                
                if result:
                    st.markdown("---")
                    
                    # Results display
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.subheader("📊 Prediction Results")
                        
                        predicted_sentiment = result['predicted_sentiment']
                        max_confidence = max(result['confidence_scores'].values())
                        
                        # Color based on sentiment
                        color_map = {"Positive": "green", "Neutral": "orange", "Negative": "red"}
                        color = color_map[predicted_sentiment]
                        
                        st.markdown(f"**Predicted Sentiment:** :{color}[{predicted_sentiment}]")
                        st.markdown(f"**Confidence:** {max_confidence:.3f}")
                        
                        if max_confidence >= confidence_threshold:
                            st.success(f"Confidence above threshold! (> {confidence_threshold})")
                        else:
                            st.warning(f"Low confidence prediction (< {confidence_threshold})")
                        
                        st.markdown(f"**Processed Text:** {result['processed_text']}")
                    
                    with col2:
                        st.subheader("📈 Confidence Scores")
                        fig = create_confidence_chart(result['confidence_scores'])
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("📁 Batch Analysis")
        st.markdown("Upload file CSV yang berisi text untuk dianalisa secara bersamaan.")
        
        with open("resource/sample.csv", "rb") as f:
            sample_csv_data = f.read()
        
        st.download_button(
            label="Download Sample CSV",
            data=sample_csv_data,
            file_name="sample.csv",
            mime="text/csv"
        )

        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type="csv",
            help=f"Maximum {MAX_BATCH_SIZE} rows allowed"
        )
        
        if uploaded_file is not None:
            try:
                string_data = uploaded_file.getvalue().decode('utf-8')
                df = pd.read_csv(StringIO(string_data))
            except UnicodeDecodeError:
                string_data = uploaded_file.getvalue().decode('cp1252')
                df = pd.read_csv(StringIO(string_data))
                
                if len(df) > MAX_BATCH_SIZE:
                    st.error(f"File has {len(df)} rows. Maximum allowed is {MAX_BATCH_SIZE} rows.")
                    st.stop()
                
                st.success(f"File uploaded, found {len(df)} rows.")
                
                # Show column
                text_column = st.selectbox(
                    "Select text column:",
                    df.columns.tolist(),
                    help="Select the column containing the texts to analyze"
                )
                
                # Show preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                analyze_batch_button = st.button(
                    "🔍 Analyze All Texts",
                    type="primary",
                    use_container_width=True
                )
                
                if analyze_batch_button:
                    texts = df[text_column].dropna().tolist()
                    
                    if not texts:
                        st.error("No valid texts found in selected column.")
                        st.stop()
                    
                    st.info(f"Analyzing {len(texts)} texts...")
                    
                    with st.spinner("Processing batch analysis..."):
                        results = predict_batch_sentiment(texts, model, tokenizer, max_length)
                    
                    if results:
                        results_df = pd.DataFrame(results)
                                                
                        # Distribution chart
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("Sentiment Distribution")
                            fig = create_batch_distribution_chart(results_df)
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("Summary Statistics")
                            sentiment_counts = results_df['predicted_sentiment'].value_counts()
                            for sentiment, count in sentiment_counts.items():
                                percentage = (count / len(results_df)) * 100
                                st.metric(sentiment, f"{count} ({percentage:.1f}%)")
                        
                        # Results table
                        st.subheader("Detailed Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="💾 Download Results as CSV",
                            data=csv,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # === Footer ===
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Indonesian Stock Sentiment Analysis | Model: <a href="https://huggingface.co/RichTan/IndoStockSentiment" target="_blank">RichTan/IndoStockSentiment</a> | Github: <a href="https://github.com/RichardDeanTan/Indonesian-Stock-Sentiment-Analysis" target="_blank">@RichardDeanTan</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()