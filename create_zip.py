import os
import zipfile

def create_kaggle_zip():
    folders_to_include = ['client', 'common', 'data', 'fault_tolerance', 'lb', 'llm', 'master', 'monitoring', 'rag', 'workers']
    files_to_include = ['config.py', 'main.py', 'requirements.txt', 'README.md']
    
    with zipfile.ZipFile('llm_cluster_code_fixed.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in folders_to_include:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if '__pycache__' in root or file.endswith('.pyc'):
                        continue
                    file_path = os.path.join(root, file)
                    # Force forward slashes for Linux/Kaggle compatibility
                    arcname = file_path.replace(os.sep, '/')
                    zipf.write(file_path, arcname)
                    
        for file in files_to_include:
            zipf.write(file, file)
            
if __name__ == '__main__':
    create_kaggle_zip()
    print("Fixed zip file created successfully!")
