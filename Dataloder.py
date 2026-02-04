"""
Einfacher Dataloader:
- listet alle Sprachdateien unter data/names/ auf,
- erzeugt One-Hot-Vektoren fuer Labels,
- konvertiert eingelesene Namen in ASCII (Akzententfernung),
- baut daraus Beispiel-Paare (Name, Label) fuer Trainings-/Testsamples.
Alle Kommentare erklaeren nur; der Code selbst bleibt unveraendert.
"""

import os  # Dateisystem-Operationen fuer Pfade und Verzeichnisse
import unicodedata  # Unicode-Normalisierung, um Akzente zu entfernen
import random  # fuer Zufallssamples aus den geladenen Daten

# Basisverzeichnis, in dem die Sprachdateien (eine pro Sprache) liegen.
path = 'data/names/'
# Alle vorhandenen Sprach-Dateinamen (z. B. "English.txt") werden ermittelt.
Langs = os.listdir(path)

# Sammelcontainer fuer alle geladenen Beispiele.
# Format jedes Elements: (name_ascii: str, onehot_label: List[int])
Datas = []


def label_to_onehot(label):
    """
    Erzeugt einen One-Hot-Vektor fuer das gegebene Sprachlabel.
    Bricht mit Fehlermeldung ab, wenn das Label nicht existiert.
    """
    if label in Langs:
        # Leerer Vektor mit einer Position pro bekannter Sprache.
        out = [0 for i in range(len(Langs))]
        # Setzt die Position des aktuellen Labels auf 1.
        out[Langs.index(label)] = 1
        return out
    # Fallback: Label ist nicht vorhanden -> abbrechen.
    print('Error Sprache nicht in Daten.')
    quit()

def saple(batch_size):
    """
    Liefert ein zufaellig gezogenes Batch aus Datas zurueck.
    - Auswahl erfolgt mit Zuruecklegen (random.choices).
    - Rueckgabe wird transponiert: erst Liste der Namen, dann Liste der Labels.
    """
    saples = random.choices(Datas, k=batch_size)
    return list(zip(*saples))


def name_to_ascii(name: str) -> str:
    """
    Wandelt einen einzelnen Namen zuverlaessig in ASCII um,
    indem Akzente entfernt und nicht darstellbare Zeichen verworfen werden.
    """
    # Zerlegt kombinierte Zeichen (z. B. "e mit Akzent") in Basiszeichen + Akzent.
    normalized = unicodedata.normalize('NFKD', name)
    # Encode/Decode entfernt alle Zeichen, die nicht im ASCII vorkommen.
    ascii_only = normalized.encode('ascii', 'ignore').decode('ascii')
    # Mehrfache Leerzeichen entfernen; Bindestriche/Spaces bleiben erhalten.
    return ' '.join(ascii_only.split())


# Debug-/Demo-Teil: Fuer jede Sprachdatei ein One-Hot-Label und Beispielnamen ausgeben.
for l in Langs:
    # Datei zeilenweise mit UTF-8 lesen, damit alle Sprachen korrekt geladen werden.
    with open(path + l, encoding='utf-8') as d:
        # Beispiel: nur die ersten drei Zeilen ausgeben, um die Ausgabe klein zu halten.
        for idx, line in enumerate(d):
            # Jede Zeile enthaelt genau einen Namen; strip() entfernt Umbrueche/Spaces.
            # name_to_ascii: Unicode -> ASCII; label_to_onehot: Zielvektor passend zur Datei.
            Datas.append((name_to_ascii(line.strip()), label_to_onehot(l.strip())))

if __name__ == '__main__':
    # Beim Direktaufruf: ziehe ein einziges Beispiel und gib es aus.
    print(saple(1))
