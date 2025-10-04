import re
from src.utils import limpiar_texto, extraer_bigrams, frases_para_resumen


def test_limpiar_texto_basic():
    s = "¡La comida del restaurante fue muy buena y el servicio excelente!"
    out = limpiar_texto(s)
    assert 'comida' in out
    assert out == out.lower()


def test_extraer_bigrams_simple():
    texts = ["muy buen servicio", "servicio muy amable", "muy buen sabor"]
    bigrams = extraer_bigrams(texts, top_n=5)
    assert isinstance(bigrams, list)
    if bigrams:
        assert isinstance(bigrams[0][0], str)
        assert isinstance(bigrams[0][1], (int, float))


def test_frases_para_resumen():
    bigrams = [('muy bueno', 3), ('buen servicio', 2), ('muy amable', 1)]
    frases = frases_para_resumen(bigrams, top_k=2)
    assert len(frases) == 2
    assert "muy bueno" in frases[0] or "buen servicio" in frases[0]
