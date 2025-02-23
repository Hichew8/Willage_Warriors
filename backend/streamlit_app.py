import streamlit as st
import pandas as pd
from ocr import analyzeImageAndExtractText
from llama_integration import get_intro_prompt, handle_user_input, send_llama_request, format_prompt, init_llama


# Set page configuration
st.set_page_config(
    page_title="Caduceo - Medical Assistant",
    page_icon="🩺",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .stButton button {
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stMarkdown h1 {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("Caduceo 🩺")
st.markdown("**Your AI-powered medical assistant**")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="padding: 10px; border-radius: 10px;">
        <h3>CPT Code Lookup 🏷️</h3>
        <p>Find medical codes quickly and accurately.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="padding: 10px; border-radius: 10px;">
        <h3>Insurance Rates 💰</h3>
        <p>Get estimated rates for procedures.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="padding: 10px; border-radius: 10px;">
        <h3>OCR Text Extraction 🔍</h3>
        <p>Extract text from images effortlessly.</p>
    </div>
    """, unsafe_allow_html=True)



# Initialize LLAMA model
if 'llama_model' not in st.session_state:
    st.session_state.llama_model = init_llama()

# Initialize conversation
if "conversation" not in st.session_state:
    st.session_state.conversation = [{"role": "assistant", "content": get_intro_prompt()}]

# Initialize state variables
if "state" not in st.session_state:
    st.session_state.state = None
if "cpt_code" not in st.session_state:
    st.session_state.cpt_code = None
if "quoted_rate" not in st.session_state:  
    st.session_state.quoted_rate = None
if "insurance" not in st.session_state:
    st.session_state.insurance = None

# Define steps
steps = ["Step 1: Enter State", "Step 2: Enter CPT Code", "Step 3: Enter Quoted Rate", "Step 4: Enter Insurance Provider", "Review Results"]

# Initialize session state for tracking progress
if "current_step" not in st.session_state:
    st.session_state.current_step = steps[0]  # Start with the first step

# Function to update the current step
def update_step():
    if st.session_state.state is None:
        st.session_state.current_step = steps[0]  # Step 1: Enter State
    elif st.session_state.cpt_code is None:
        st.session_state.current_step = steps[1]  # Step 2: Enter CPT Code
    elif st.session_state.quoted_rate is None:
        st.session_state.current_step = steps[2]  # Step 3: Enter Quoted Rate
    elif st.session_state.insurance is None:
        st.session_state.current_step = steps[3]  # Step 4: Enter Insurance Provider
    else:
        st.session_state.current_step = steps[4]  # Step 5: Review Results

# Update the step based on conversation progress
update_step()

with st.sidebar:
    st.header("Caduceo Sidebar 🛠️")
    st.markdown("Customize your chat experience here.")
    st.header("Progress Tracker 📊")
    st.write(f"**{st.session_state.current_step}**")
    st.progress(steps.index(st.session_state.current_step) / 4)
    if st.button("Choose a file 📁"):
        st.session_state.show_uploader = True
    if st.button("Clear Conversation 🗑️"):
        st.session_state.state = None
        st.session_state.insurance = None
        st.session_state.cpt_code = None
        st.session_state.quoted_rate = None
        st.session_state.conversation = []
        st.session_state.conversation = [{"role": "assistant", "content": get_intro_prompt()}]
        st.success("Conversation cleared!")
    st.header("Export Data 💾")
    if st.button("Download Conversation as Text 📄"):
        chat_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.conversation])
        st.download_button(
            label="Download",
            data=chat_history,
            file_name="conversation.txt",
            mime="text/plain"
        )
    st.header("About ℹ️")
    st.markdown("""
    **Caduceo** is an AI-powered medical assistant designed to help healthcare professionals and patients with:
    - Medical code lookup
    - Insurance rate estimation
    - OCR-based text extraction
    """)
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: gray;">
        Made by the Willage Warriors Team | <a href="#">Privacy Policy</a> | <a href="#">Contact Us</a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: gray;">
        Version ℹ️: 1.2.0
    </div>
    """, unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.conversation:
    if (message["role"] == "user" and 
    message["content"].startswith("Extracted OCR text:")):
        continue
    with st.chat_message(message["role"]):
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"])
        else:
            st.markdown(message["content"])

# OCR Image Uploader Integration
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False

if st.session_state.get("show_uploader", False):
    uploaded_image = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        st.success("File uploaded successfully! 🎉")
        with st.spinner("Extracting text from image... 🔍"):
            ocr_result = analyzeImageAndExtractText(uploaded_image)
        
        st.markdown(f"**Image Caption:** {ocr_result['caption']}")
        
        ocr_message = f"Extracted OCR text:\n{ocr_result['extractedText']}"
        st.session_state.conversation.append({"role": "user", "content": ocr_message})
        
        st.success("OCR text has been added to the conversation! 🎉")
        st.session_state.show_uploader = False

# Handle user input
handle_user_input()