
import os
import streamlit as st
import base64
from openai import OpenAI

# Función para codificar la imagen en base64
def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode("utf-8")


# Configuración de la página
st.set_page_config(
    page_title="Análisis de imagen",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==========================================================
# DISEÑO VISUAL PERSONALIZADO (CSS)
# ==========================================================
st.markdown("""
<style>
/* Fondo general */
.stApp {
    background: linear-gradient(
        135deg,
        #101827 0%,
        #17243A 55%,
        #101827 100%
    );
    color: #F8FAFC;
}

/* Contenedor principal */
.block-container {
    max-width: 950px;
    padding-top: 2.5rem;
    padding-bottom: 3rem;
}

/* Título */
h1 {
    color: #FFFFFF !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    text-align: center;
    letter-spacing: -1px;
    padding-bottom: 1rem;
}

/* Encabezados */
h2, h3, h4 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
}

/* Texto general y etiquetas */
.stApp p,
.stApp label,
.stMarkdown,
.stCaption,
[data-testid="stWidgetLabel"] p {
    color: #E2E8F0 !important;
}

/* Campo de API Key */
.stTextInput input,
.stTextArea textarea {
    background-color: #0B1220 !important;
    color: #FFFFFF !important;
    border: 1px solid #64748B !important;
    border-radius: 10px !important;
    caret-color: #FFFFFF;
}

/* Texto de ayuda */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #CBD5E1 !important;
    opacity: 1 !important;
}

/* Enfoque de los campos */
.stTextInput input:focus,
.stTextArea textarea:focus {
    border: 1px solid #38BDF8 !important;
    box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.2) !important;
}

/* Botones principales */
.stButton > button {
    background: linear-gradient(
        90deg,
        #0284C7,
        #2563EB
    ) !important;
    color: #FFFFFF !important;
    border: 1px solid #60A5FA !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.5rem !important;
    font-weight: 700 !important;
    min-height: 45px;
    transition: all 0.2s ease-in-out;
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

/* Interruptor */
[data-testid="stToggle"] label,
[data-testid="stToggle"] p {
    color: #F8FAFC !important;
}

/* Paneles desplegables */
[data-testid="stExpander"] {
    background-color: #1E293B !important;
    border: 1px solid #475569 !important;
    border-radius: 14px !important;
    overflow: hidden;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #FFFFFF !important;
    font-weight: 600 !important;
}

/* Imagen */
[data-testid="stImage"] {
    background-color: #0B1220;
    padding: 10px;
    border: 1px solid #475569;
    border-radius: 14px;
}

/* Cargador de archivos */
[data-testid="stFileUploader"] {
    background-color: #1E293B;
    border: 1px dashed #64748B;
    border-radius: 14px;
    padding: 15px;
}

[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
    color: #F1F5F9 !important;
}

/* Botón del cargador */
[data-testid="stFileUploader"] button {
    background-color: #334155 !important;
    color: #FFFFFF !important;
    border: 1px solid #94A3B8 !important;
    border-radius: 8px !important;
}

/* Área de respuesta */
[data-testid="stMarkdownContainer"] {
    color: #F1F5F9;
    line-height: 1.75;
}

/* Texto en negrita de las respuestas */
[data-testid="stMarkdownContainer"] strong {
    color: #FFFFFF !important;
}

/* Alertas y mensajes */
[data-testid="stAlert"] {
    border-radius: 12px;
    border: 1px solid #64748B;
}

[data-testid="stAlert"] p {
    color: #FFFFFF !important;
}

/* Indicador de carga */
[data-testid="stSpinner"] p {
    color: #FDE68A !important;
    font-weight: 600;
}

/* Separadores */
hr {
    border-color: #475569 !important;
}

/* Selección de texto */
::selection {
    background-color: #0369A1;
    color: #FFFFFF;
}

/* Diseño adaptable a móviles */
@media (max-width: 640px) {
    .block-container {
        padding-top: 1.2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    h1 {
        font-size: 1.8rem !important;
    }

    .stButton > button {
        width: 100%;
    }
}
</style>
""", unsafe_allow_html=True)


# ==========================================================
# APLICACIÓN ORIGINAL
# ==========================================================

st.title("Análisis de Imagen: 🤖🏞️")

ke = st.text_input(
    "Ingresa tu Clave",
    type="password",
    placeholder="Introduce tu API key de OpenAI"
)

os.environ["OPENAI_API_KEY"] = ke

# Recuperar API Key
api_key = os.environ["OPENAI_API_KEY"]

# Inicializar cliente de OpenAI
client = OpenAI(api_key=api_key) if api_key else None

# Carga de imágenes
uploaded_file = st.file_uploader(
    "Sube una imagen",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    with st.expander("Imagen cargada", expanded=True):
        st.image(
            uploaded_file,
            caption=uploaded_file.name,
            use_container_width=True
        )

# Interruptor para contexto adicional
show_details = st.toggle(
    "Pregunta algo específico sobre la imagen",
    value=False
)

# Inicializar el contexto para evitar variables no definidas
additional_details = ""

if show_details:
    additional_details = st.text_area(
        "Añade contexto o escribe tu pregunta aquí:",
        placeholder="Ejemplo: ¿Qué objetos aparecen en el fondo?",
        disabled=not show_details
    )

# Botón de análisis
analyze_button = st.button(
    "🔍 Analiza la imagen",
    type="secondary",
    use_container_width=True
)

# Validación y análisis
if analyze_button:

    if not uploaded_file:
        st.warning("Por favor, sube una imagen.")

    elif not api_key:
        st.warning("Por favor, ingresa tu API key.")

    else:
        with st.spinner("Analizando la imagen..."):

            # Codificar la imagen
            base64_image = encode_image(uploaded_file)

            # Preparar el prompt
            prompt_text = (
                "Describe lo que ves en la imagen en español."
            )

            if show_details and additional_details.strip():
                prompt_text += (
                    "\n\nContexto o pregunta adicional del usuario:\n"
                    + additional_details
                )

            # Detectar el formato real de la imagen
            mime_type = uploaded_file.type or "image/jpeg"

            # Preparar los mensajes
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": (
                                    f"data:{mime_type};base64,"
                                    f"{base64_image}"
                                )
                            }
                        }
                    ]
                }
            ]

            # Solicitud a OpenAI y respuesta en streaming
            try:
                full_response = ""
                message_placeholder = st.empty()

                for completion in client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    max_tokens=1200,
                    stream=True
                ):
                    if completion.choices[0].delta.content is not None:
                        full_response += (
                            completion.choices[0].delta.content
                        )

                        message_placeholder.markdown(
                            full_response + "▌"
                        )

                # Mostrar respuesta final
                message_placeholder.markdown(full_response)

            except Exception as e:
                st.error(f"Ocurrió un error: {e}")

