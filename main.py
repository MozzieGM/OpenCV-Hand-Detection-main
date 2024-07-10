import cv2 
import numpy as np
from cvzone.HandTrackingModule import HandDetector

# Inicializa a captura de vídeo
video = cv2.VideoCapture(0)

# Define a largura e a altura da captura de vídeo
video.set(3, 1280)
video.set(4, 720)

# Inicializa o detector de mãos
detector = HandDetector()
desenho = []  # Lista para armazenar os pontos de desenho

# Carrega o classificador de rosto pré-treinado
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carrega a imagem PNG que você quer usar para cobrir o rosto
overlay_img = cv2.imread('overlay.png', cv2.IMREAD_UNCHANGED)  # Carrega a imagem com suporte a canal alfa

# Define um fator de escala para aumentar o tamanho da imagem
scale_factor = 2.0  # Aumente este valor para aumentar a imagem

def overlay_image(bg_img, overlay_img, pos):
    """Overlay `overlay_img` on top of `bg_img` at the position specified by `pos`."""
    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(bg_img.shape[0], y + overlay_img.shape[0])
    x1, x2 = max(0, x), min(bg_img.shape[1], x + overlay_img.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(overlay_img.shape[0], bg_img.shape[0] - y)
    x1o, x2o = max(0, -x), min(overlay_img.shape[1], bg_img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    alpha_mask = overlay_img[:, :, 3] / 255.0  # Normaliza o canal alfa para 0-1
    alpha_composite = 1.0 - alpha_mask

    for c in range(0, 3):
        bg_img[y1:y2, x1:x2, c] = (alpha_mask[y1o:y2o, x1o:x2o] * overlay_img[y1o:y2o, x1o:x2o, c] +
                                   alpha_composite[y1o:y2o, x1o:x2o] * bg_img[y1:y2, x1:x2, c])

while True:
    check, img = video.read()  # Lê um frame do vídeo
    resultado = detector.findHands(img, draw=True)  # Detecta mãos no frame
    hand = resultado[0]  # Obtém a informação da primeira mão detectada

    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detecta rostos na imagem
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenha um retângulo sobre cada rosto detectado
    for (x, y, w, h) in faces:
        # Redimensiona a imagem de sobreposição para corresponder ao tamanho do rosto, mas com o fator de escala
        overlay_resized = cv2.resize(overlay_img, (int(w * scale_factor), int(h * scale_factor)))

        # Calcula a nova posição para centralizar a imagem redimensionada
        new_x = x - (overlay_resized.shape[1] - w) // 2
        new_y = y - (overlay_resized.shape[0] - h) // 2

        # Sobrepõe a imagem na posição do rosto
        overlay_image(img, overlay_resized, (new_x, new_y))

    if hand:
        lmlist = hand[0]['lmList']  # Lista de coordenadas dos pontos de referência da mão
        dedos = detector.fingersUp(hand[0])  # Verifica quais dedos estão levantados
        dedosLev = dedos.count(1)  # Conta quantos dedos estão levantados

        if dedosLev == 1:
            # Desenha um círculo na ponta do dedo indicador e adiciona a posição à lista de desenho
            x, y = lmlist[8][0], lmlist[8][1]
            cv2.circle(img, (x, y), 15, (0, 0, 255), cv2.FILLED)
            desenho.append((x, y))
        elif dedosLev != 1 and dedosLev != 3:
            # Adiciona um ponto "nulo" (0, 0) para interromper o desenho
            desenho.append((0, 0))
        elif dedosLev == 3:
            # Limpa a lista de desenho
            desenho = []

        # Desenha linhas conectando os pontos na lista de desenho
        for id, ponto in enumerate(desenho):
            x, y = ponto[0], ponto[1]
            cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)  # Altere a cor do círculo para verde
            if id >= 1:
                ax, ay = desenho[id-1][0], desenho[id-1][1]
                if x != 0 and ax != 0:
                    cv2.line(img, (x, y), (ax, ay), (0, 255, 0), 20)  # Altere a cor da linha para verde

    # Redimensiona a imagem para aumentar o tamanho da janela
    img_resized = cv2.resize(img, (1280, 720))  # Aqui você define a nova resolução da janela

    cv2.imshow('Mozzie', img_resized)  # Exibe a imagem redimensionada
    if cv2.waitKey(1) == 27:  # Sai do loop se a tecla 'ESC' for pressionada
        break

# Libera o vídeo e destrói todas as janelas OpenCV
video.release()
cv2.destroyAllWindows()
