import base64
import binascii
from enum import Enum

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(
    title="Image Processing API",
    description="API для создания размытых подложек для изображений.",
    version="1.0.0"
)

# Параметры размытия
BLUR_KERNEL_SIZE = (21, 21)  # ТОЛЬКО нечетные числа. Чем больше - тем сильнее размытие
CROP_PIXELS = 3  # количество пикселей, которые мы обрезаем по периметру оригинального изображения (чтобы избежать
# черной рамки)


class AspectRatio(str, Enum):
    one_to_one = "1:1"
    seven_to_ten = "7:10"


class ResizeRequest(BaseModel):
    image_base64: str
    aspect_ratio: AspectRatio


def create_cover(original_image: np.ndarray, aspect_ratio: AspectRatio) -> np.ndarray:
    if aspect_ratio == AspectRatio.one_to_one:
        return blur_cover_1_1(original_image)
    elif aspect_ratio == AspectRatio.seven_to_ten:
        return blur_cover_7_10(original_image)


def blur_cover_1_1(original_image: np.ndarray) -> np.ndarray:
    """
    Функция для создания отблюренной подложки для формата 1:1
    """
    # выбираем размер изображения
    h, w, _ = original_image.shape
    new_size = max(h, w, 512)

    # создаем размытый фон
    resized_for_blur = cv2.resize(original_image, (new_size, new_size), interpolation=cv2.INTER_AREA)
    small_img = cv2.resize(resized_for_blur, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    blurred_small = cv2.GaussianBlur(small_img, BLUR_KERNEL_SIZE, 0)
    blurred_background = cv2.resize(blurred_small, (new_size, new_size), interpolation=cv2.INTER_CUBIC)

    # Обрезка и вставка оригинала в центр
    image_to_paste = original_image
    # Проверяем, достаточно ли велико изображение для обрезки
    if h > CROP_PIXELS * 2 and w > CROP_PIXELS * 2:
        image_to_paste = original_image[CROP_PIXELS:-CROP_PIXELS, CROP_PIXELS:-CROP_PIXELS]
    paste_h, paste_w, _ = image_to_paste.shape

    y_offset = (new_size - paste_h) // 2
    x_offset = (new_size - paste_w) // 2
    blurred_background[y_offset:y_offset + paste_h, x_offset:x_offset + paste_w] = image_to_paste
    return blurred_background


def blur_cover_7_10(original_image: np.ndarray) -> np.ndarray:
    """
    Функция для создания отблюренной подложки для формата 1:1
    """
    # выбираем размер изображения
    h, w, _ = original_image.shape
    target_ratio = 7.0 / 10.0

    if w / h > target_ratio:
        new_width = w
        new_height = int(w / target_ratio)
    else:
        new_height = h
        new_width = int(h * target_ratio)

    if new_width < 616 or new_height < 880:
        new_width = 616
        new_height = 880

    # создаем размытый фон
    resized_for_blur = cv2.resize(original_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    small_img = cv2.resize(resized_for_blur, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    blurred_small = cv2.GaussianBlur(small_img, BLUR_KERNEL_SIZE, 0)
    blurred_background = cv2.resize(blurred_small, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    # Обрезка и вставка оригинала в центр
    image_to_paste = original_image
    if h > CROP_PIXELS * 2 and w > CROP_PIXELS * 2:
        image_to_paste = original_image[CROP_PIXELS:-CROP_PIXELS, CROP_PIXELS:-CROP_PIXELS]

    paste_h, paste_w, _ = image_to_paste.shape

    y_offset = (new_height - paste_h) // 2
    x_offset = (new_width - paste_w) // 2

    blurred_background[y_offset:y_offset + paste_h, x_offset:x_offset + paste_w] = image_to_paste

    return blurred_background


@app.get("/health", summary="Проверка работоспособности сервиса")
async def health_check():
    return JSONResponse(content={"status": "OK"}, status_code=200)


@app.post("/resize", summary="Изменение размера и создание подложки")
async def resize_image(request: ResizeRequest):
    try:
        image_bytes = base64.b64decode(request.image_base64)
        image_array = np.frombuffer(image_bytes, np.uint8)
        original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if original_image is None:
            raise HTTPException(status_code=400, detail="Invalid image data. Could not decode image.")
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="Invalid base64 string.")

    processed_image_array = create_cover(original_image, request.aspect_ratio)
    _, buffer = cv2.imencode('.jpg', processed_image_array)
    processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

    return JSONResponse(
        content={
            "processed_image_base64": processed_image_base64,
            "format": "jpeg"
        }
    )


"""
Для локального тестирования REST API и работы ручек
"""
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
