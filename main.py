import subprocess
import os
import platform
import urllib.request

def run_command(command, shell=False):
    """Run a system command, handles exceptions."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while executing {command}: {e}")

def install_libcublas():
    """Install libcublas (only applicable for Linux)"""
    if platform.system() == "Linux":
        print("Installing libcublas on Linux...")
        run_command(["sudo", "apt", "install", "-y", "libcublas11"])
    else:
        print("libcublas installation is skipped on non-Linux systems.")

def clone_repo(repo_url, folder_name):
    """Clone a repository using git."""
    if not os.path.exists(folder_name):
        run_command(["git", "clone", repo_url])
    else:
        print(f"Repository {folder_name} already exists, skipping clone.")

def install_requirements(requirements_file):
    """Install Python packages from requirements.txt"""
    run_command(["pip", "install", "-r", requirements_file])

def download_file(url, output_path):
    """Download a file from the internet."""
    try:
        print(f"Downloading {output_path}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"Downloaded to {output_path}")
    except Exception as e:
        print(f"Error downloading {output_path}: {e}")

def run_python_module(module, cwd):
    """Run a Python module from a given directory."""
    run_command(["python", "-m", module], shell=True)

def main():
    # Step 1: Install libcublas (only for Linux)
    install_libcublas()

    # Step 2: Clone GenEHR repository
    clone_repo("https://github.com/logesh-works/GenEHR", "GenEHR")

    # Step 3: Install Python requirements for GenEHR
    install_requirements("GenEHR/requirements.txt")

    # Step 4: Clone NeMo repository, checkout the nemo-v2 branch, and run reinstall.sh
    clone_repo("https://github.com/AI4Bharat/NeMo.git", "NeMo")
    
    # Change directory to NeMo, checkout nemo-v2 branch and run reinstall.sh
    os.chdir("NeMo")
    run_command(["git", "checkout", "nemo-v2"])
    
    # Check platform and use shell command if bash is required
    if platform.system() == "Windows":
        print("For Windows, ensure you have Git Bash installed to run bash scripts.")
        run_command(["bash", "reinstall.sh"], shell=True)
    else:
        run_command(["bash", "reinstall.sh"])
    
    os.chdir("..")  # Return to the original directory

    # Step 5: Download the checkpoint.nemo file
    download_file("https://objectstore.e2enetworks.net/indic-asr-public/indicConformer/ai4b_indicConformer_ta.nemo", "checkpoint.nemo")

    # Step 6: Run the GenEHR Python module
    os.chdir("GenEHR")
    run_python_module("GenEHR", "GenEHR")

if __name__ == "__main__":
    main()
