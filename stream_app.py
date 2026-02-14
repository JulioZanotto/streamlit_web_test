import streamlit as st
import requests
import base64
import json
import time
import io
from PIL import Image, ImageDraw
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(layout="wide", page_title="Teste de Inferência YOLOv11")

# --- Configurações Laterais ---
st.sidebar.header("Configurações AWS")
# Tenta pegar dos secrets do Streamlit, se não achar, fica vazio (ou lança erro)
try:
    API_KEY = st.secrets["AWS_API_KEY"]
    API_URL = st.secrets["AWS_URL"]
except FileNotFoundError:
    st.error("Chave de API não configurada nos Secrets!")
    st.stop()
CONCURRENCY = st.sidebar.slider("Requisições Simultâneas", 1, 50, 5)

st.title("🚀 Teste de Carga e Inferência Reconhecimento de motos")

uploaded_files = st.file_uploader("Escolha as imagens", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

def process_image(uploaded_file):
    """Envia uma única imagem para a API e retorna o resultado"""
    try:
        # Preparar Imagem
        image_bytes = uploaded_file.getvalue()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {"image": base64_image}
        headers = {"Content-Type": "application/json"}
        if API_KEY:
            headers["x-api-key"] = API_KEY

        start_time = time.time()
        
        # Chamada à API
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            return {
                "file_name": uploaded_file.name,
                "status": "Sucesso",
                "duration": duration,
                "data": response.json(),
                "original_image": image_bytes
            }
        else:
            return {
                "file_name": uploaded_file.name,
                "status": f"Erro {response.status_code}",
                "duration": duration,
                "error": response.text
            }
    except Exception as e:
        return {"file_name": uploaded_file.name, "status": "Erro Local", "error": str(e)}

if st.button("Enviar para Inferência") and uploaded_files:

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Processamento Paralelo (Simula carga)
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = [executor.submit(process_image, file) for file in uploaded_files]
        
        for i, future in enumerate(futures):
            result = future.result()
            results.append(result)
            progress_bar.progress((i + 1) / len(uploaded_files))
            status_text.text(f"Processando {i+1}/{len(uploaded_files)}...")

    st.success("Processamento concluído!")
    
    # --- Exibição de Resultados ---
    st.subheader("Métricas")
    tempos = [r['duration'] for r in results if 'duration' in r]
    if tempos:
        avg_time = sum(tempos) / len(tempos)
        col1, col2, col3 = st.columns(3)
        col1.metric("Tempo Médio", f"{avg_time:.3f} s")
        col2.metric("Total Imagens", len(results))
        col3.metric("Sucessos", len([r for r in results if r['status'] == "Sucesso"]))

    st.subheader("Visualização")
    cols = st.columns(3)
    for idx, res in enumerate(results):
        if res['status'] == "Sucesso":
            # 1. Debug (Opcional)
            # print(f"Retorno Lambda ({res['file_name']}):", res['data'])

            try:
                # O 'data' JÁ É o dicionário {'resultado': {...}}
                lambda_response = res['data']
                
                # --- CORREÇÃO AQUI ---
                # Acessamos direto a chave 'resultado', sem procurar por 'body'
                dados_inferencia = lambda_response.get('resultado', {})
                
                classe = dados_inferencia.get('class', 'Desconhecido')
                confidence = dados_inferencia.get('confidence', 0.0)

            except Exception as e:
                print(f"Erro ao ler dados: {e}")
                classe = "Erro Leitura"
                confidence = 0.0

            # 2. Preparar Imagem e Legenda
            img = Image.open(io.BytesIO(res['original_image']))
            
            # Formatação
            label = f"{classe.upper()} ({confidence*100:.1f}%)"
            
            # Desenhar na imagem (Opcional, mas ajuda a salvar a imagem já com tag se quiser)
            draw = ImageDraw.Draw(img)
            # Dica: Posição (10,10) pode ficar ruim se a moto estiver no canto. 
            # Mas para teste está ótimo.
            draw.text((10, 10), label, fill="red")

            with cols[idx % 3]:
                st.image(img, caption=f"{res['file_name']} - {label}", use_container_width=True)
        else:
            st.error(f"Falha em {res['file_name']}: {res.get('error')}")
