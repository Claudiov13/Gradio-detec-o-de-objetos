import gradio as gr
from ultralytics import YOLO
import numpy as np

# 1. Carrega o modelo YOLO
model = YOLO('yolov8n.pt')

# 2. Define a função que processará cada frame da webcam
def reconhecer_objetos_em_webcam(frame):
    if frame is None:
        # Retorna um frame preto se a entrada for nula
        return np.zeros((480, 640, 3), dtype=np.uint8)

    # Inicializa o frame anotado com o frame original
    # Isso garante que a imagem não fique preta se não houver detecções
    annotated_frame = frame
    
    # Roda o modelo no frame da imagem
    results = model(frame, stream=True)

    # Obtém o frame com as detecções desenhadas
    # Se houver detecções, o valor de annotated_frame será atualizado
    for r in results:
        annotated_frame = r.plot()
        
    return annotated_frame

# 3. Cria a interface do Gradio
demo = gr.Interface(
    fn=reconhecer_objetos_em_webcam,
    inputs=gr.Image(
        sources=["webcam", "upload"],
        label="Sua Webcam ou Imagem"
    ),
    outputs=gr.Image(
        label="Objetos Reconhecidos"
    ),
    title="Detecção de Objetos com YOLOv8 e Gradio",
    description="Uma demonstração de detecção de objetos em tempo real usando sua webcam ou uma imagem estática."
)

# 4. Inicia a interface
demo.launch(share=True)