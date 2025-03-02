import os
import sys
import torch
from torch.utils.data import DataLoader
from PIL import Image

from model import VAE, vae_loss, Generator, Discriminator
from dataset import ImageDataset

# Funkcja montowania Google Drive – działa w Google Colab
def mount_drive_if_colab():
    try:
        import google.colab
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive zamontowany.")
    except ImportError:
        print("Nie uruchamiasz w środowisku Colab.")

# Przykładowa funkcja generująca dummy model 3D (kostka)
def generate_dummy_3d_model():
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
    mount_drive_if_colab()
    
    image_path = input("Podaj pełną ścieżkę do pliku ze zdjęciem 4K (np. /content/drive/MyDrive/4K_image.jpg): ").strip()
    if not os.path.exists(image_path):
        print("Podany plik nie istnieje! Sprawdź ścieżkę.")
        sys.exit(1)
    
    dataset = ImageDataset(image_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for img in dataloader:
        input_img = img.to("cuda" if torch.cuda.is_available() else "cpu")
        print("Obraz wczytany. Rozmiary:", input_img.size())
    
    latent_dim = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Inicjalizacja VAE i optymalizatora
    vae = VAE(latent_dim).to(device)
    optimizer_vae = torch.optim.Adam(vae.parameters(), lr=0.0002)
    
    num_epochs = 10
    for epoch in range(num_epochs):
        optimizer_vae.zero_grad()
        recon_img, mu, logvar = vae(input_img.to(device))
        loss = vae_loss(recon_img, input_img.to(device), mu, logvar)
        loss.backward()
        optimizer_vae.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.2f}")
    
    os.makedirs("outputs", exist_ok=True)
    output_path = os.path.join("outputs", "reconstructed.jpg")
    recon_np = recon_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
    recon_np = (recon_np * 255).clip(0,255).astype('uint8')
    Image.fromarray(recon_np).save(output_path)
    print("Rekonstrukcja zapisana w:", output_path)
    
    # Eksport przykładowego modelu 3D do .glb
    from export_3d import export_model_to_glb
    model_data = generate_dummy_3d_model()
    export_model_to_glb(model_data["vertices"], model_data["faces"], filename="outputs/model.glb")
    
    print("Projekt uruchomiony pomyślnie.")

if __name__ == "__main__":
    main()
