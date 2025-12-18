from fastapi import APIRouter
from pydantic import BaseModel
import torch
import os
from pathlib import Path

from app.services.machine_translator import MachineTranslator, MachineTranslatorConfig

router = APIRouter()

class TranslateRequest(BaseModel):
    text: str
    direction: str  # "vi-to-en" | "en-to-vi" (mở rộng sau)

class TranslateResponse(BaseModel):
    translated_text: str

def load_translator():
    BASE_DIR = Path(__file__).resolve().parent.parent
    # -> backend/app

    weights_path = BASE_DIR / "services" / "translator_weights" / "weights.pth"

    translator = MachineTranslator(MachineTranslatorConfig)
    translator.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=True)
    )
    translator.eval()
    return translator



translator = load_translator()


@router.post("/translate", response_model=TranslateResponse)
def translate_api(payload: TranslateRequest):
    # Hiện tại bạn chỉ hỗ trợ vi -> en
    translated_sentence = translator.translate(payload.text, 2)

    return {
        "translated_text": translated_sentence
    }
