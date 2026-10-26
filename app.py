
import os
import json
import re
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# =====================================================
# DISEÑO
# =====================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: #f3f6fc;
}
[data-testid="stSidebar"] {
    background: #14213d;
}
[data-testid="stSidebar"] * {
    color: white;
}
.hero {
    background: linear-gradient(120deg, #14213d, #315b9b);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 34px;
}
.hero p {
    color: #e0eaff;
}
.card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #dfe7f3;
    margin-bottom: 15px;
}
div.stButton > button {
    border-radius: 10px;
    font-weight: bold;
    min-height: 42px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📚 DocuMind AI</h1>
    <p>Tu asistente inteligente para analizar documentos PDF.</p>
    <p>💬 Chat · 📝 Resúmenes · 🧠 Cuestionarios</p>
</div>
""", unsafe_allow_html=True)

st.write("Versión de Python:", platform.python_version())

try:
    image = Image.open("Chat_pdf.png")
    st.image(image, width=250)
except Exception:
    pass

# =====================================================
# BARRA LATERAL ORIGINAL
# =====================================================

with st.sidebar:
    st.subheader("📄 Analizador inteligente de documentos")
    st.write("Carga un PDF y analiza su contenido.")

ke = st.text_input("Ingresa tu Clave de OpenAI", type="password")

if ke:
    os.environ["OPENAI_API_KEY"] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

pdf = st.file_uploader("📂 Carga el archivo PDF", type="pdf")


# =====================================================
# ESTADO DE SESIÓN
# =====================================================

for key, default in {
    "pdf_hash": None,
    "texto": "",
    "fragmentos": [],
    "base": None,
    "resumen": None,
    "preguntas": [],
    "resultado": None,
    "respuestas": {},
    "historial": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# =====================================================
# FUNCIONES NUEVAS
# =====================================================

def generar_resumen(fragmentos):
    modelo = OpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo-instruct"
    )

    resumenes = []

    instrucciones = """
    Resume el texto en español.
    Incluye las ideas principales, los conceptos importantes
    y las conclusiones. No inventes información.
    """

    for inicio in range(0, len(fragmentos), 6):
        bloque = "\n\n".join(fragmentos[inicio:inicio + 6])
        resumen = modelo.predict(
            instrucciones + "\n\nDocumento:\n" + bloque
        )
        resumenes.append(resumen)

    if len(resumenes) == 1:
        return resumenes[0]

    return modelo.predict(
        "Combina estos resúmenes en uno solo, claro y organizado. "
        "No repitas ideas ni inventes información:\n\n"
        + "\n\n".join(resumenes)
    )


def generar_cuestionario(fragmentos, cantidad):
    modelo = OpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo-instruct"
    )

    contenido = "\n\n".join(fragmentos[:35])

    instrucciones = f"""
    Genera exactamente {cantidad} preguntas de selección múltiple
    basadas únicamente en el documento.

    Cada pregunta debe tener cuatro opciones: A, B, C y D,
    una respuesta correcta y una explicación breve.

    Devuelve exclusivamente JSON válido con esta estructura:
    {{
      "preguntas": [
        {{
          "pregunta": "Pregunta",
          "opciones": {{
            "A": "Opción A",
            "B": "Opción B",
            "C": "Opción C",
            "D": "Opción D"
          }},
          "correcta": "A",
          "explicacion": "Explicación"
        }}
      ]
    }}
    """

    resultado = modelo.predict(
        instrucciones + "\n\nDOCUMENTO:\n" + contenido
    ).strip()

    resultado = re.sub(r"^```(?:json)?\s*", "", resultado)
    resultado = re.sub(r"\s*```$", "", resultado)

    datos = json.loads(resultado)
    preguntas = datos.get("preguntas", [])

    validas = []

    for pregunta in preguntas:
        opciones = pregunta.get("opciones", {})
        correcta = pregunta.get("correcta", "")

        if (
            pregunta.get("pregunta")
            and all(letra in opciones for letra in "ABCD")
            and correcta in ["A", "B", "C", "D"]
        ):
            validas.append(pregunta)

    if not validas:
        raise ValueError("No se generaron preguntas válidas.")

    return validas


# =====================================================
# PROCESAMIENTO DEL PDF ORIGINAL
# =====================================================

if pdf is not None and ke:
    try:
        contenido_pdf = pdf.getvalue()
        import hashlib
        hash_actual = hashlib.sha256(contenido_pdf).hexdigest()

        if hash_actual != st.session_state["pdf_hash"]:
            from io import BytesIO

            pdf_reader = PdfReader(BytesIO(contenido_pdf))
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text() or ""

            if not text.strip():
                st.error(
                    "No se encontró texto en el PDF. "
                    "Si es escaneado, necesitas OCR."
                )
                st.stop()

            st.info(f"Texto extraído: {len(text)} caracteres")

            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=500,
                chunk_overlap=20,
                length_function=len
            )

            chunks = text_splitter.split_text(text)

            st.success(f"Documento dividido en {len(chunks)} fragmentos")

            embeddings = OpenAIEmbeddings()
            knowledge_base = FAISS.from_texts(chunks, embeddings)

            st.session_state.update({
                "pdf_hash": hash_actual,
                "texto": text,
                "fragmentos": chunks,
                "base": knowledge_base,
                "resumen": None,
                "preguntas": [],
                "resultado": None,
                "respuestas": {},
                "historial": []
            })

        text = st.session_state["texto"]
        chunks = st.session_state["fragmentos"]
        knowledge_base = st.session_state["base"]

        # =================================================
        # ESTADÍSTICAS VISUALES
        # =================================================

        st.markdown("## 📊 Información del documento")

        c1, c2, c3 = st.columns(3)

        c1.metric("📄 Páginas", len(PdfReader(pdf).pages))
        c2.metric("🧩 Fragmentos", len(chunks))
        c3.metric("🔤 Caracteres", f"{len(text):,}")

        # =================================================
        # PESTAÑAS
        # =================================================

        tab_chat, tab_resumen, tab_quiz = st.tabs([
            "💬 Chat con PDF",
            "📝 Resumen automático",
            "🧠 Cuestionario"
        ])

        # =================================================
        # CHAT ORIGINAL
        # =================================================

        with tab_chat:
            st.subheader("💬 Pregunta sobre tu documento")

            with st.form("form_chat"):
                user_question = st.text_area(
                    "Escribe tu pregunta",
                    placeholder="¿Cuáles son las ideas principales?"
                )

                enviar = st.form_submit_button(
                    "🔎 Consultar documento",
                    type="primary"
                )

            if enviar and user_question.strip():
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI(
                    temperature=0,
                    model_name="gpt-3.5-turbo-instruct"
                )

                chain = load_qa_chain(llm, chain_type="stuff")

                response = chain.run(
                    input_documents=docs,
                    question=user_question
                )

                st.session_state["historial"].append({
                    "pregunta": user_question,
                    "respuesta": response
                })

            for mensaje in reversed(st.session_state["historial"]):
                with st.container(border=True):
                    st.markdown(f"**🙋 Pregunta:** {mensaje['pregunta']}")
                    st.markdown("**🤖 Respuesta:**")
                    st.markdown(mensaje["respuesta"])

            if st.session_state["historial"]:
                if st.button("🗑️ Limpiar conversación"):
                    st.session_state["historial"] = []
                    st.rerun()

        # =================================================
        # RESUMEN AUTOMÁTICO
        # =================================================

        with tab_resumen:
            st.markdown("""
            <div class="card">
                <h3>📝 Resumen inteligente</h3>
                <p>Descubre las ideas principales y conclusiones
                de tu documento sin leerlo por completo.</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button(
                "✨ Generar resumen del PDF",
                type="primary",
                use_container_width=True
            ):
                try:
                    with st.spinner("Analizando el documento..."):
                        st.session_state["resumen"] = generar_resumen(chunks)
                except Exception as e:
                    st.error(f"Error generando resumen: {e}")

            if st.session_state["resumen"]:
                st.markdown("### 📖 Resultado")
                st.markdown(st.session_state["resumen"])

                st.download_button(
                    "⬇️ Descargar resumen",
                    data=st.session_state["resumen"],
                    file_name="resumen_pdf.txt",
                    mime="text/plain"
                )

        # =================================================
        # GENERADOR DE CUESTIONARIOS
        # =================================================

        with tab_quiz:
            st.markdown("""
            <div class="card">
                <h3>🧠 Pon a prueba tus conocimientos</h3>
                <p>Genera preguntas basadas en el PDF,
                responde y recibe tu puntuación.</p>
            </div>
            """, unsafe_allow_html=True)

            cantidad = st.selectbox(
                "Cantidad de preguntas",
                [3, 5, 10],
                index=1
            )

            if st.button(
                "🎯 Generar cuestionario",
                type="primary",
                use_container_width=True
            ):
                try:
                    with st.spinner("Creando preguntas..."):
                        st.session_state["preguntas"] = generar_cuestionario(
                            chunks, cantidad
                        )
                        st.session_state["resultado"] = None
                        st.session_state["respuestas"] = {}
                except Exception as e:
                    st.error(f"Error generando cuestionario: {e}")

            preguntas = st.session_state["preguntas"]

            if preguntas:
                with st.form("form_quiz"):
                    respuestas = {}

                    for i, pregunta in enumerate(preguntas):
                        st.markdown(f"#### Pregunta {i + 1}")
                        st.write(pregunta["pregunta"])

                        opciones = pregunta["opciones"]

                        respuestas[i] = st.radio(
                            "Elige una respuesta",
                            ["A", "B", "C", "D"],
                            format_func=lambda letra, ops=opciones:
                                f"{letra}. {ops[letra]}",
                            key=f"quiz_{hash_actual}_{i}",
                            index=None
                        )

                        st.divider()

                    calificar = st.form_submit_button(
                        "✅ Calificar cuestionario",
                        type="primary",
                        use_container_width=True
                    )

                if calificar:
                    if any(v is None for v in respuestas.values()):
                        st.warning("Responde todas las preguntas.")
                    else:
                        aciertos = sum(
                            respuestas[i] == p["correcta"]
                            for i, p in enumerate(preguntas)
                        )

                        st.session_state["respuestas"] = respuestas
                        st.session_state["resultado"] = {
                            "aciertos": aciertos,
                            "total": len(preguntas)
                        }

                resultado = st.session_state["resultado"]

                if resultado:
                    porcentaje = round(
                        resultado["aciertos"] /
                        resultado["total"] * 100
                    )

                    st.markdown("### 🏆 Resultados")
                    col1, col2 = st.columns(2)

                    col1.metric("Puntuación", f"{porcentaje}%")
                    col2.metric(
                        "Respuestas correctas",
                        f"{resultado['aciertos']} de {resultado['total']}"
                    )

                    st.progress(porcentaje / 100)

                    if porcentaje >= 80:
                        st.success("¡Excelente trabajo!")
                    elif porcentaje >= 60:
                        st.info("Buen trabajo. Sigue practicando.")
                    else:
                        st.warning("Repasa el documento e inténtalo de nuevo.")

                    st.markdown("### 📋 Revisión de respuestas")

                    for i, pregunta in enumerate(preguntas):
                        respuesta = st.session_state["respuestas"][i]
                        correcta = pregunta["correcta"]

                        with st.expander(
                            f"{'✅' if respuesta == correcta else '❌'} "
                            f"Pregunta {i + 1}"
                        ):
                            st.write(pregunta["pregunta"])
                            st.write(
                                f"Tu respuesta: {respuesta}. "
                                f"{pregunta['opciones'][respuesta]}"
                            )
                            st.write(
                                f"Respuesta correcta: {correcta}. "
                                f"{pregunta['opciones'][correcta]}"
                            )
                            st.write(pregunta.get("explicacion", ""))

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")

elif pdf is not None and not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

else:
    st.info("📂 Carga un archivo PDF para comenzar.")

st.markdown("---")
st.caption("📚 DocuMind AI | Chat, resúmenes y cuestionarios")


