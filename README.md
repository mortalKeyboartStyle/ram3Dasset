# ram-3Dasset

Projekt dedykowany do generowania modeli 3D (.glb) z obrazów 4K.  
Projekt jest RAM-centric i oparty o hybrydowe podejście, łączące modele VAE, GAN oraz moduł eksportu 3D.  
Można go uruchomić zarówno lokalnie (np. w VS Code), jak i w Google Colab z integracją Google Drive.

## Struktura projektu

- `main.py` – Główny skrypt uruchamiający projekt.
- `model.py` – Definicje modeli: VAE, GAN.
- `dataset.py` – Klasa dataset do wczytywania obrazu 4K.
- `export_3d.py` – Moduł eksportu modelu 3D do formatu .glb.
- `requirements.txt` – Lista zależności.
- `.gitignore` – Lista plików ignorowanych przez Git.
- `README.md` – Dokumentacja projektu.

## Uruchomienie lokalne

1. Utwórz wirtualne środowisko:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
