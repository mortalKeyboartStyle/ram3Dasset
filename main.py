import os
import sys
import torch
from torch.utils.data import DataLoader
from PIL import Image

from model import VAE, vae_loss, Generator, Discriminator, reparameterize
from dataset import ImageDataset

# Funkcja montowania Google Drive – działa tylko w Colabie
def mount_drive_if_colab():
    try:
        import google.colab
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive zamontowany.")
    except Exception as e:
        print("Nie uruchamiasz w środowisku Colab, pomijam montowanie Drive.")

def generate_dummy_3d_model():
    # Przykładowy model 3D – kostka
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1]
    ]
    faces = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [1, 2, 6],
        [1, 6, 5],
        [0, 3, 7],
        [0, 7, 4]
    ]
    return {"vertices": vertices, "faces": faces}

def main():
    # Montowanie Google Drive (jeśli uruchamiasz w Colab)
    mount_drive_if_colab()
    
    # Pobranie ścieżki do pliku ze zdjęciem od użytkownika
    image_path = input("Podaj pełną ścieżkę do pliku ze zdjęciem 4K (np. /content/drive/MyDrive/4K_image.jpg): ").strip()
    if not os.path.exists(image_path):
        print("Podany plik nie istnieje! Sprawdź ścieżkę:", image_path)
        sys.exit(1)
    
    # Wczytanie obrazu przy użyciu datasetu
    dataset = ImageDataset(image_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for img in dataloader:
        input_img = img.to("cuda" if torch.cuda.is_available() else "cpu")
        print("Obraz wczytany. Rozmiary:", input_img.size())
    
    # Ustawienia modelu
    latent_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inicjalizacja modelu VAE i optymalizatora
    vae = VAE(latent_dim).to(device)
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=0.0002)
    
    # Trening modelu VAE (przykładowa pętla na 10 epok)
    num_epochs = 10
    for epoch in range(num_epochs):
        optimizer_vae.zero_grad()
        recon_img, mu, logvar = vae(input_img.to(device))
        loss = vae_loss(recon_img, input_img.to(device), mu, logvar)
        loss.backward()
        optimizer_vae.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}")
    
    # Ustalenie ścieżki zapisu wyników
    # Jeśli uruchamiasz w Colabie, zapisz na Google Drive; lokalnie – w folderze outputs/
    if os.path.exists("/content/drive/MyDrive"):
        output_dir = "/content/drive/MyDrive/ram3Dasset_outputs"
    else:
        output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Zapis rekonstrukcji obrazu
    output_recon = os.path.join(output_dir, "reconstructed.jpg")
    recon_np = recon_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    recon_np = (recon_np * 255).clip(0, 255).astype('uint8')
    Image.fromarray(recon_np).save(output_recon)
    print("Rekonstrukcja zapisana w:", output_recon)
    
    # Eksport przykładowego modelu 3D do pliku .glb
    from export_3d import export_model_to_glb
    model_data = generate_dummy_3d_model()
    output_model = os.path.join(output_dir, "model.glb")
    export_model_to_glb(model_data["vertices"], model_data["faces"], filename=output_model)
    
    print("Projekt uruchomiony pomyślnie.")

if __name__ == "__main__":
    main()
