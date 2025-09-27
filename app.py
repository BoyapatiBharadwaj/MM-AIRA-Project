import streamlit as st
from core.nlp_processor import NLPProcessor
from core.rag_pipeline import RAGPipeline
from core.sentiment import SentimentEmotionAnalyzer
from core.explainable import TextExplainer
from utils.file_utils import read_pdf, read_text_file
from PIL import Image
import torch
import numpy as np
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
# -------------------- CV Utilities --------------------

st.set_page_config(page_title="MM-AIRA", layout="wide")
st.title("MM-AIRA: Multi-Modal AI Research Assistant")

mode = st.sidebar.selectbox("Select Input Type", ["Text / PDF", "Image"])

# -------------------- Multilingual Translation --------------------
def translate_text(text, src="auto", tgt="en"):
    from transformers import MarianMTModel, MarianTokenizer
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
    except:
        return text
    translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# -------------------- YOLO Object Detection --------------------
def yolo_detect(img_array):
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")  # small model
    results = model.predict(img_array)
    annotated_img = results[0].plot()
    return annotated_img

# -------------------- BLIP Image QA --------------------
def blip_image_qa(image, question):
    from transformers import BlipProcessor, BlipForQuestionAnswering
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    inputs = processor(images=image, text=question, return_tensors="pt")
    out = model.generate(**inputs)
    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer

# -------------------- Image Explainable AI --------------------
def image_explanation(img_array):
    from captum.attr import IntegratedGradients
    import torch
    img_tensor = torch.tensor(img_array.transpose(2,0,1)/255.0, dtype=torch.float).unsqueeze(0)
    ig = IntegratedGradients(torch.nn.Identity())  # placeholder model
    attr, _ = ig.attribute(img_tensor, target=0, return_convergence_delta=True)
    attr = attr.detach().cpu().numpy()[0].transpose(1,2,0)
    attr = np.uint8(255 * (attr - attr.min()) / (attr.max() - attr.min() + 1e-8))
    return attr

# -------------------- Text / PDF Mode --------------------
if mode == "Text / PDF":
    uploaded_file = st.file_uploader("Upload PDF or Text file", type=["pdf","txt"])
    query = st.text_input("Ask a question or leave empty for summary")
    translate = st.checkbox("Translate to English (if multilingual)")

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            text = read_pdf(uploaded_file)
        else:
            text = read_text_file(uploaded_file)

        if translate:
            text = translate_text(text, tgt="en")

        st.subheader("Document Text Preview")
        st.write(text[:1000] + "..." if len(text) > 1000 else text)

        # NLP + RAG
        rag = RAGPipeline(llm_model="llama2:7b")  # smaller LLM for laptop
        if query:
            try:
                answer = rag.query(text, query)
                st.subheader("RAG Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"RAG LLM failed: {e}")
        else:
            nlp = NLPProcessor()
            chunks = nlp.split_text(text)
            st.subheader("Document Summary (First 2 Chunks)")
            for r in chunks[:2]:
                st.write(r)

        # Sentiment & Emotion
        analyzer = SentimentEmotionAnalyzer()
        sentiment = analyzer.analyze_sentiment(text)
        emotion = analyzer.analyze_emotion(text)

        st.subheader("Sentiment Analysis")
        st.write(sentiment)
        st.subheader("Emotion Analysis")
        st.write(emotion)

        # Explainable AI (Text)
        explainer = TextExplainer("distilbert-base-uncased-finetuned-sst-2-english")
        try:
            attributions = explainer.explain(text)
            st.subheader("Text Explanation (Feature Attributions)")
            st.write(attributions)
        except:
            st.write("Text explainability not available.")

# -------------------- Image Mode --------------------
elif mode == "Image":
    uploaded_file = st.file_uploader("Upload an Image", type=["png","jpg","jpeg"])
    question = st.text_input("Ask a question about the image (optional)")
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_array = np.array(image)

        # Basic CV: edges
        cv_proc = CVProcessor()
        edges = cv_proc.simple_object_detection(img_array)

        st.subheader("Original Image")
        st.image(image, use_column_width=True)
        st.subheader("Edges Detection")
        st.image(edges, use_column_width=True)

        # YOLO Object Detection
        st.subheader("YOLOv8 Object Detection")
        try:
            annotated_img = yolo_detect(img_array)
            st.image(annotated_img, use_column_width=True)
        except Exception as e:
            st.error(f"YOLO detection failed: {e}")

        # BLIP Image QA
        if question:
            try:
                answer = blip_image_qa(image, question)
                st.subheader("BLIP Image QA Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"BLIP QA failed: {e}")

        # Image Explainable AI
        st.subheader("Image Explanation (Grad-CAM Approx)")
        try:
            attr_img = image_explanation(img_array)
            st.image(attr_img, use_column_width=True)
        except Exception as e:
            st.error(f"Image explanation failed: {e}")
