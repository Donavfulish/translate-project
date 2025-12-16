import torch
import os

from services.machine_translator import MachineTranslator, MachineTranslatorConfig

def translate_api(source_sentence):
    weights_path = os.path.join("services", "translator_weights", "weights.pth")
    translator = MachineTranslator(MachineTranslatorConfig)
    translator.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    
    translated_sentence = translator.translate(source_sentence, 2)
    return translated_sentence
