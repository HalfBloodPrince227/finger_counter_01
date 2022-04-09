import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)  # подключение к web-камере
mp_Hands = mp.solutions.hands  # говорим, что хотим распознать руки
hands = mp_Hands.Hands(max_num_hands = 2)  # характеристики для распознования
mpDraw = mp.solutions.drawing_utils  # инициализация утилит рисования

fingers_coord = [(8, 6), (12, 10), (16, 14), (20, 18)]  # координаты ключевых точек на руке, кроме большого пальца
thumb_coord = (4,2)  # координаты ключевых точек для большого пальца

while cap.isOpened():  # пока камера "работает"
    success, image = cap.read()  # получаем кадр с камеры
    if not success:  # если не удалось получить кадр с камеры
        print('Не удалось получить кадр с web-камеры')
        continue  # переход к ближайшему циклу (while)
    image = cv2.flip(image, 1)  # зеркально отражаем изображение
    RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR -> RGB изображение
    results = hands.process(RGB_image)  # ищем руки на изображении 
    multiLandMarks = results.multi_hand_landmarks  # извлекаем список найденных рук
    
    if multiLandMarks:  # если руки найдены
        upCount = 0
        for idx, handLms in enumerate(multiLandMarks):  # перебираем найденные руки
            lbl = results.multi_handedness[idx].classification[0].label
            print(lbl)
    
            mpDraw.draw_landmarks(image, handLms, mp_Hands.HAND_CONNECTIONS)
            fingersList = []
            for idx, Lm in enumerate(handLms):  # перебираем ключевые точки на руке
                h, w, c = image.shape
                cx, cy = int(Lm.x * w), int(Lm.y * h)
                fingersList.append(cx, cy)
            for coordinate in fingers_coord:
                if fingersList[coordinate[0]][1] < fingersList[coordinate[1]][1]:
                    upCount += 1
        print(upCount)
   
   
    cv2.imshow('web-cam', image)
    if cv2.waitKey(1) & 0xFF == 27:  # ожидаем нажатия клавиши Esc
        break

cap.release()