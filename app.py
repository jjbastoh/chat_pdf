
import os
import json
import hashlib
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS


st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide"
)

# ---------------- DISEÑO ----------------
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
    background: linear-gradient(120deg,#14213d,#315b9b);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}
.hero h1 {
    color: white;
    font-size: 35px;
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
    <p>Analiza tus documentos con inteligencia artificial.</p>
    <p>💬 Chat con PDF · 📝 Resúmenes · 🧠 Cuestionarios</p>
</div>
""", unsafe_allow_html=True)


# ---------------- ESTADO ----------------
for key, value in {
    "pdf_hash": None,
    "texto": "",
    "fragmentos": [],
    "paginas": 0,
    "base": None,
    "resumen": None,
    "quiz": [],
    "resultado": None,
    "historial": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- BARRA LATERAL ----------------
with st.sidebar:
    st.title("⚙️ Configuración")
    api_key = st.text_input(
        "Clave API de OpenAI",
        type="password"
    )

    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.markdown("---")
    st.markdown("### 🛠️ Herramientas")
    st.markdown("""
    - 📄 Análisis de documentos
    - 💬 Chat inteligente
    - 📝 Resumen automático
    - 🧠 Generador de cuestionarios
    """)

    if st.session_state["texto"]:
        st.markdown("---")
        st.markdown("### 📊 Documento")
        st.write(f"Páginas: {st.session_state['paginas']}")
        st.write(
            f"Fragmentos: {len(st.session_state['fragmentos'])}"
        )


# ---------------- CARGAR PDF ----------------
st.markdown("## 📄 Carga tu documento")

pdf = st.file_uploader(
    "Selecciona un archivo PDF",
    type=["pdf"]
)


def llamar_ia(instrucciones, contenido):
    modelo = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )
    respuesta = modelo.invoke([
        ("system", instrucciones),
        ("human", contenido)
    ])
    return respuesta.content


def crear_resumen(fragmentos):
    instrucciones = """
    Resume el documento en español.
    Identifica las ideas principales, conceptos importantes
    y conclusiones. No inventes información.
    """

    resumenes = []

    for i in range(0, len(fragmentos), 8):
        bloque = "\n\n".join(fragmentos[i:i + 8])
        resumen = llamar_ia(instrucciones, bloque)
        resumenes.append(resumen)

    if len(resumenes) == 1:
        return resumenes[0]

    return llamar_ia(
        "Combina estos resúmenes en uno solo, claro y organizado. "
        "Elimina repeticiones y no inventes información.",
        "\n\n".join(resumenes)
    )


def crear_quiz(fragmentos, cantidad):
    contenido = "\n\n".join(fragmentos[:40])

    instrucciones = f"""
    Crea exactamente {cantidad} preguntas de selección múltiple
    basadas únicamente en el documento.

    Cada pregunta tendrá cuatro opciones A, B, C y D,
    una respuesta correcta y una explicación.

    Devuelve únicamente JSON válido con esta estructura:
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

    resultado = llamar_ia(instrucciones, contenido).strip()

    if resultado.startswith("```"):
        resultado = resultado.split("\n", 1)[1]
        resultado = resultado.rsplit("```", 1)[0].strip()

    datos = json.loads(resultado)
    preguntas = datos.get("preguntas", [])

    validas = []
    for p in preguntas:
        if (
            p.get("pregunta")
            and all(k in p.get("opciones", {}) for k in "ABCD")
            and p.get("correcta") in "ABCD"
            and len(p.get("correcta", "")) == 1
        ):
            validas.append(p)

    if not validas:
        raise ValueError("La IA no generó preguntas válidas.")

    return validas


if pdf and api_key:
    try:
        datos_pdf = pdf.getvalue()
        hash_actual = hashlib.sha256(datos_pdf).hexdigest()

        if hash_actual != st.session_state["pdf_hash"]:
            with st.spinner("📖 Procesando el documento..."):
                import io

                lector = PdfReader(io.BytesIO(datos_pdf))
                texto = "\n".join(
                    pagina.extract_text() or ""
                    for pagina in lector.pages
                )

                if not texto.strip():
                    st.error(
                        "No se encontró texto en el PDF. "
                        "Si está escaneado, necesitas OCR."
                    )
                    st.stop()

                divisor = CharacterTextSplitter(
                    separator="\n",
                    chunk_size=500,
                    chunk_overlap=20,
                    length_function=len
                )
                fragmentos = divisor.split_text(texto)

                embeddings = OpenAIEmbeddings()
                base = FAISS.from_texts(fragmentos, embeddings)

                st.session_state.update({
                    "pdf_hash": hash_actual,
                    "texto": texto,
                    "fragmentos": fragmentos,
                    "paginas": len(lector.pages),
                    "base": base,
                    "resumen": None,
                    "quiz": [],
                    "resultado": None,
                    "historial": []
                })

            st.success("Documento procesado correctamente.")

    except Exception as e:
        st.error(f"Error procesando el PDF: {e}")


# ---------------- HERRAMIENTAS ----------------
if st.session_state["texto"] and api_key:

    c1, c2, c3 = st.columns(3)
    c1.metric("📄 Páginas", st.session_state["paginas"])
    c2.metric("🧩 Fragmentos", len(st.session_state["fragmentos"]))
    c3.metric("🔤 Caracteres", f"{len(st.session_state['texto']):,}")

    chat_tab, resumen_tab, quiz_tab = st.tabs([
        "💬 Chat con PDF",
        "📝 Resumen automático",
        "🧠 Cuestionario"
    ])

    # ---------------- CHAT RAG ----------------
    with chat_tab:
        st.markdown("### 💬 Pregunta a tu documento")

        with st.form("chat_form"):
            pregunta = st.text_area(
                "¿Qué quieres saber?",
                placeholder="Escribe aquí tu pregunta..."
            )
            enviar = st.form_submit_button(
                "🔎 Consultar",
                type="primary"
            )

        if enviar and pregunta.strip():
            try:
                with st.spinner("Buscando respuesta..."):
                    documentos = st.session_state[
                        "base"
                    ].similarity_search(pregunta, k=4)

                    contexto = "\n\n".join(
                        doc.page_content for doc in documentos
                    )

                    respuesta = llamar_ia(
                        "Responde en español usando únicamente "
                        "el contexto del documento. Si no hay "
                        "información suficiente, dilo claramente.",
                        f"Contexto:\n{contexto}\n\nPregunta: {pregunta}"
                    )

                    st.session_state["historial"].append(
                        (pregunta, respuesta)
                    )

            except Exception as e:
                st.error(f"Error en la consulta: {e}")

        for pregunta_anterior, respuesta_anterior in reversed(
            st.session_state["historial"]
        ):
            with st.container(border=True):
                st.markdown(f"**🙋 Pregunta:** {pregunta_anterior}")
                st.markdown(f"**🤖 Respuesta:** {respuesta_anterior}")

        if st.session_state["historial"]:
            if st.button("🗑️ Limpiar historial"):
                st.session_state["historial"] = []
                st.rerun()

    # ---------------- RESUMEN ----------------
    with resumen_tab:
        st.markdown("### 📝 Resumen inteligente")
        st.write(
            "Obtén las ideas principales, los conceptos clave "
            "y las conclusiones del documento."
        )

        if st.button(
            "✨ Generar resumen",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner("Analizando el documento..."):
                    st.session_state["resumen"] = crear_resumen(
                        st.session_state["fragmentos"]
                    )
            except Exception as e:
                st.error(f"No se pudo generar el resumen: {e}")

        if st.session_state["resumen"]:
            st.markdown("---")
            st.markdown("#### 📖 Resultado")
            st.markdown(st.session_state["resumen"])

            st.download_button(
                "⬇️ Descargar resumen",
                data=st.session_state["resumen"],
                file_name="resumen.txt",
                mime="text/plain"
            )

    # ---------------- CUESTIONARIO ----------------
    with quiz_tab:
        st.markdown("### 🧠 Pon a prueba tus conocimientos")
        st.write(
            "Genera preguntas de selección múltiple basadas "
            "en el contenido del PDF."
        )

        cantidad = st.selectbox(
            "Número de preguntas",
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
                    st.session_state["quiz"] = crear_quiz(
                        st.session_state["fragmentos"],
                        cantidad
                    )
                    st.session_state["resultado"] = None
            except Exception as e:
                st.error(f"Error creando el cuestionario: {e}")

        preguntas = st.session_state["quiz"]

        if preguntas:
            with st.form("quiz_form"):
                respuestas = {}

                for i, p in enumerate(preguntas):
                    st.markdown(f"#### Pregunta {i + 1}")
                    st.write(p["pregunta"])

                    respuestas[i] = st.radio(
                        "Selecciona una respuesta",
                        ["A", "B", "C", "D"],
                        format_func=lambda x, opciones=p["opciones"]:
                            f"{x}. {opciones[x]}",
                        key=f"respuesta_{st.session_state['pdf_hash']}_{i}",
                        index=None
                    )

                    st.divider()

                calificar = st.form_submit_button(
                    "✅ Calificar respuestas",
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
                    st.session_state["resultado"] = {
                        "respuestas": respuestas,
                        "aciertos": aciertos,
                        "total": len(preguntas)
                    }

            resultado = st.session_state["resultado"]

            if resultado:
                porcentaje = round(
                    resultado["aciertos"] / resultado["total"] * 100
                )

                st.markdown("### 🏆 Resultado")
                st.metric("Puntuación", f"{porcentaje}%")
                st.progress(porcentaje / 100)
                st.write(
                    f"Respuestas correctas: "
                    f"{resultado['aciertos']} de {resultado['total']}"
                )

                for i, p in enumerate(preguntas):
                    respuesta = resultado["respuestas"][i]
                    correcta = p["correcta"]

                    with st.expander(f"Pregunta {i + 1} — "
                                     f"{'✅' if respuesta == correcta else '❌'}"):
                        st.write(p["pregunta"])
                        st.write(
                            f"Tu respuesta: {respuesta}. "
                            f"{p['opciones'][respuesta]}"
                        )
                        st.write(
                            f"Correcta: {correcta}. "
                            f"{p['opciones'][correcta]}"
                        )
                        st.write(p.get("explicacion", ""))

else:
    st.info(
        "🔐 Ingresa tu clave de OpenAI y carga un PDF "
        "para comenzar el análisis."
    )

st.markdown("---")
st.caption("📚 DocuMind AI · Chat, resúmenes y cuestionarios con IA")

