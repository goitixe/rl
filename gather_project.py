import os

# Carpetas y archivos a ignorar para no ensuciar
IGNORE_DIRS = {'venv', '.git', '__pycache__', 'logs', 'models', '.idea', '.vscode'}
IGNORE_FILES = {'gather_project.py', '.DS_Store'}

def gather_code():
    project_root = os.getcwd()
    output = []
    
    print(f"--- RECOPILANDO PROYECTO EN: {project_root} ---\n")
    
    for root, dirs, files in os.walk(project_root):
        # Filtrar carpetas ignoradas
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file in IGNORE_FILES:
                continue
            
            # Solo nos interesan estos formatos
            if file.endswith(('.py', '.md', '.txt')):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, project_root)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        output.append(f"\n{'='*20} FILE: {rel_path} {'='*20}\n")
                        output.append(content)
                        output.append(f"\n{'='*50}\n")
                except Exception as e:
                    print(f"Error leyendo {rel_path}: {e}")

    # Imprimir todo el resultado junto
    full_text = "".join(output)
    print(full_text)
    
    # Opcional: Guardarlo en un archivo para copiar más fácil
    with open("project_context.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    print("\n\n--- LISTO: Todo el contenido se ha guardado en 'project_context.txt' ---")

if __name__ == "__main__":
    gather_code()