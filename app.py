import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# ==========================================================
# DISEÑO VISUAL PERSONALIZADO: AZUL OSCURO Y AZUL ELÉCTRICO
# ==========================================================

st.markdown("""
<style>

/* Fondo principal */
.stApp {
    background: linear-gradient(
        135deg,
        #0F172A 0%,
        #172554 55%,
        #0F172A 100%
    );
    color: #F8FAFC;
}

/* Contenedor principal */
.block-container {
    max-width: 1050px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Título principal */
h1 {
    color: #FFFFFF !important;
    font-size: 2.35rem !important;
    font-weight: 800 !important;
    text-align: center;
    letter-spacing: -0.7px;
    padding-bottom: 1rem;
}

/* Encabezados */
h2, h3, h4 {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Texto general */
.stApp p,
.stApp label,
.stMarkdown,
.stCaption,
[data-testid="stWidgetLabel"] p {
    color: #F1F5F9 !important;
}

/* Barra lateral */
[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827 0%,
        #1E293B 100%
    );
    border-right: 1px solid #475569;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
}

/* Campos de texto */
.stTextInput input,
.stTextArea textarea {
    background-color: #0B1220 !important;
    color: #FFFFFF !important;
    border: 1px solid #64748B !important;
    border-radius: 12px !important;
    caret-color: #FFFFFF !important;
}

/* Texto de ejemplo */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #CBD5E1 !important;
    opacity: 1 !important;
}

/* Enfoque de campos */
.stTextInput input:focus,
.stTextArea textarea:focus {
    border: 1px solid #38BDF8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}

/* Botones */
.stButton > button {
    background: linear-gradient(
        90deg,
        #0284C7,
        #2563EB
    ) !important;
    color: #FFFFFF !important;
    border: 1px solid #60A5FA !important;
    border-radius: 12px !important;
    min-height: 44px;
    font-weight: 700 !important;
    transition: all 0.2s ease;
}

/* Texto de los botones */
.stButton > button p {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* Efecto hover */
.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #0369A1,
        #1D4ED8
    ) !important;
    border-color: #BFDBFE !important;
    transform: translateY(-2px);
}

/* Cargador de archivos PDF */
[data-testid="stFileUploader"] {
    background-color: #1E293B !important;
    border: 1px dashed #94A3B8 !important;
    border-radius: 14px !important;
    padding: 15px !important;
}

/* Texto del cargador */
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    color: #F8FAFC !important;
}

/* Botón para seleccionar archivo */
[data-testid="stFileUploader"] button {
    background-color: #334155 !important;
    color: #FFFFFF !important;
    border: 1px solid #94A3B8 !important;
    border-radius: 8px !important;
}

/* Imagen del encabezado */
[data-testid="stImage"] {
    background-color: #111827;
    padding: 10px;
    border: 1px solid #475569;
    border-radius: 14px;
}

/* Mensajes informativos */
[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid #64748B;
}

[data-testid="stAlert"] p {
    color: #FFFFFF !important;
}

/* Respuestas */
[data-testid="stMarkdownContainer"] {
    color: #F1F5F9 !important;
    line-height: 1.8;
}

/* Texto importante */
[data-testid="stMarkdownContainer"] strong {
    color: #FFFFFF !important;
}

/* Código y bloques de texto */
.stCodeBlock,
[data-testid="stCode"] {
    border: 1px solid #475569;
    border-radius: 10px;
}

/* Separadores */
hr {
    border-color: #475569 !important;
}

/* Tablas */
[data-testid="stTable"] {
    color: #FFFFFF !important;
}

/* Adaptación para móviles */
@media (max-width: 640px) {
    .block-container {
        padding: 1.2rem 1rem 2rem 1rem;
    }

    h1 {
        font-size: 1.75rem !important;
    }

    .stButton > button {
        width: 100%;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# APLICACIÓN RAG ORIGINAL
# ==========================================================

# Título y presentación
st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.write("Versión de Python:", platform.python_version())

# Cargar y mostrar imagen
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Información en la barra lateral
with st.sidebar:
    st.subheader(
        "Este Agente te ayudará a realizar análisis sobre el PDF cargado"
    )

# Obtener clave API
ke = st.text_input(
    'Ingresa tu Clave de OpenAI',
    type="password"
)

if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning(
        "Por favor ingresa tu clave de API de OpenAI para continuar"
    )

# Cargador de archivos PDF
pdf = st.file_uploader(
    "Carga el archivo PDF",
    type="pdf"
)

# Procesar PDF si se ha cargado
if pdf is not None and ke:
    try:
        # Extraer texto del PDF
        pdf_reader = PdfReader(pdf)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        st.info(f"Texto extraído: {len(text)} caracteres")

        # Dividir texto en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        st.success(
            f"Documento dividido en {len(chunks)} fragmentos"
        )

        # Crear embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Interfaz para realizar preguntas
        st.subheader(
            "Escribe qué quieres saber sobre el documento"
        )

        user_question = st.text_area(
            " ",
            placeholder="Escribe tu pregunta aquí..."
        )

        # Procesar pregunta
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(
                temperature=0,
                model_name="gpt-4o-mini-2024-07-18"
            )

            # Cargar cadena de preguntas y respuestas
            chain = load_qa_chain(
                llm,
                chain_type="stuff"
            )

            # Obtener respuesta
            response = chain.run(
                input_documents=docs,
                question=user_question
            )

            # Mostrar respuesta
            st.markdown("### Respuesta:")
            st.markdown(response)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")

        # Mostrar detalles del error para depuración
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning(
        "Por favor ingresa tu clave de API de OpenAI para continuar"
    )

else:
    st.info(
        "Por favor carga un archivo PDF para comenzar"
    )

