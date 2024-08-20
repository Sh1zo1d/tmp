
# функция для обрезки номера 
def crop_image(image, box):
    x_min, y_min, x_max, y_max = map(int, box)  
    return image[y_min:y_max, x_min:x_max]
# забираем "метаданные" из названия файла
def extract_metadata_from_filename(filename):
    parts = filename.split()
    checkpoint_number = parts[2]
    date_str = parts[3]
    video_time_start_str = parts[4].split('.')[0]
    
    date = datetime.strptime(date_str, '%Y-%m-%d')
    video_time_start = datetime.strptime(video_time_start_str, '%H-%M-%S_%f')    
    return checkpoint_number, date, video_time_start
# сколько прошло от старта
def compute_time_from_start(start_time, elapsed_seconds):
    full_date_time = start_time + timedelta(seconds=elapsed_seconds)
    return full_date_time.time()

def is_valid_plate(plate):
    # маска для проверки на допустимые символы и формат номера
    pattern = re.compile(r'^[ABEKMHOPCTYX]\d{3}[ABEKMHOPCTYX]{2}\d{2,3}$|^[ABEKMHOPCTYX]{2}\d{6,7}$', re.IGNORECASE)
    return bool(pattern.match(plate))

def save_annotation(plate, image_shape):
    annotation = {
        "root": {
            "tags": [],
            "objects": [],
            "description": plate,
            "name": plate,
            "region_id": 6,
            "state_id": 2,
            "count_lines": 0,
            "size": {
                "width": image_shape[1],
                "height": image_shape[0]
            },
            "moderation": {
                "isModerated": 0
            }
        }
    }
    return annotation

def video_processing(video_path, new_lst=None):
    # создаем папки, если их нет
    img_dir = 'whisper/new_data/final_test_20_08_2024/img'
    ann_dir = 'whisper/new_data/final_test_20_08_2024/ann'
    processed_video_dir = 'whisper/new_data/final_test_20_08_2024/processed_video'
    os.makedirs(img_dir, exist_ok=True)  
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(processed_video_dir, exist_ok=True)

    filenames = new_lst
    results = []
    # confidence_threshold = yolo_confidence_threshold  # порог для yolo
# цикл проход для каждого файла из new_lst
    for filename in filenames:
        print(filename)
          # читаем видео
#         file_path = os.path.join(video_path, filename)
        cap = cv2.VideoCapture(filename)
        # проверка успешного открытия видео
        if not cap.isOpened():
            print(f"Ошибка открытия {file_path}")
            exit()
        # получаем данные о fps в видео
        # поулчаем FPS w h видео
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = 0
      

        output_filename = 'processed' +'_'+filename
        processed_video_path = os.path.join(processed_video_dir,output_filename)
# достаем метаданные и иную информацию из названия файла
      
        checkpoint_number, date, video_start_time = extract_metadata_from_filename(filename)
      # настройка VideoWriter для сохранения выходного видео
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))
      # создание словаря для хранения истории треков объектов
        track_history = defaultdict(lambda: [])
      # пробегаем по каждому кадру
        while True:
            ret, frame = cap.read()
          # если не можем прочесть кадр, то выходим из цикла
            if not ret:
                break

            frame_count += 1
          # обрабатываем только каждый 10 файл
            if frame_count % 10 == 0:
                time_from_start_in_sec = frame_count / fps

                # предсказываем боксы номерных пластин при помощи Yolo
                yolo_pred = model_od.track(frame, verbose=False, persist=True)
                if yolo_pred[0].boxes is not None and yolo_pred[0].boxes.id is not None:
                # если уверенность модели меньше порога - откидываем
                    boxes_xyxy = yolo_pred[0].boxes.xyxy.cpu().numpy()
                    boxes = yolo_pred[0].boxes.xywh.cpu().numpy()
                    track_ids = yolo_pred[0].boxes.id.int().cpu().tolist()  # идентификаторы треков
                    confidences = yolo_pred[0].boxes.conf.cpu().numpy()  # уверенность в боксе
                    annotated_frame = results[0].plot()

                    if len(boxes) == 0:
                        continue
                  # отрисовка треков
                    for box, track_id in zip(boxes, track_ids):
                        x, y, w, h = box  # координаты центра и размеры бокса
                        track = track_history[track_id]
                        track.append((float(x), float(y)))  # добавление координат центра объекта в историю
                        if len(track) > 30:  # ограничение длины истории до 30 кадров
                            track.pop(0)
                    for box in boxes_xyxy:
                        cropped_image = crop_image(frame, box)
                        ocr_prediction = model_ocr(cropped_image)
  
                      # проверка распознанного номера на соответствие маскам
                        if not is_valid_plate(ocr_prediction):
                          continue  
  
  
  
  
  
                        results.append({
                              'id_plate': f'{frame_count}_{filename}',  
                              'file_path': file_path,
                              'date': date,
                              'plate_recognition': ocr_prediction,
                              'time_from_start_in_sec': time_from_start_in_sec,
                              'checkpoint_number': checkpoint_number
                          })
  # Создаем jpg и  json с предсказаниями
                          # сохраняем изображение номера
                        img_filename = os.path.join(img_dir, f"{ocr_prediction}.jpg")
                        cv2.imwrite(img_filename, cropped_image)
  
                          # сохраняем аннотацию
                        ann_filename = os.path.join(ann_dir, f"{ocr_prediction}.json")
                        annotation = save_annotation(ocr_prediction, cropped_image.shape)
                        with open(ann_filename, 'w') as ann_file:
                            json.dump(annotation, ann_file, ensure_ascii=False, indent=4)
  
                    
                          # отрисовка линий трека
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
    
                          # визуализация результатов на кадре
                        annotated_frame = results[0].plot()
                  
                          # отрисовка треков и идентификаторов
                        for box, track_id, confidence in zip(boxes, track_ids, confidences):
                            x, y, w, h = box  # координаты центра и размеры бокса
                  
                            # бокс и номер
                            x1, y1, x2, y2 = int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated_frame, f"ID: {track_id} Conf: {confidence:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    out.write(annotated_frame)  # запись кадра в выходное видео
            else:
                out.write(frame)  # запись кадра в выходное видео
                
        cap.release()
    
    # итоговы датафрейм 
    results_df = pd.DataFrame(results, columns=['id_plate', 'file_path', 'date', 'plate_recognition', 'time_from_start_in_sec', 'checkpoint_number'])
    return results_df
