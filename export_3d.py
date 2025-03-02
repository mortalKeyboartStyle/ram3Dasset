import trimesh

def export_model_to_glb(vertices, faces, filename="model.glb"):
    """
    Eksportuje model 3D do formatu .glb przy użyciu biblioteki trimesh.
    :param vertices: Lista wierzchołków
    :param faces: Lista ścian (indeksów wierzchołków)
    :param filename: Nazwa pliku wyjściowego (.glb)
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.export(filename)
    print(f"Model 3D zapisany jako {filename}")
