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


def create_voices_directory():
    """Create a 'voices' directory if it doesn't exist."""
    voices_dir = os.path.join(os.getcwd(), "voices")
    if not os.path.exists(voices_dir):
        os.makedirs(voices_dir)
        print(f"Directory 'voices' created at {voices_dir}.")
    else:
        print(f"Directory 'voices' already exists at {voices_dir}.")

def main():
    # Step 1: Install libcublas (only for Linux)
    install_libcublas()
    
    create_voices_directory()

    # Step 4: Install Python requirements for GenEHR
    install_requirements("requirements.txt")

    # Step 5: Clone NeMo repository, checkout the nemo-v2 branch, and run reinstall.sh
    clone_repo("https://github.com/logesh-works/NeMo.git", "NeMo")
    
    # Change directory to NeMo, checkout master branch and run reinstall.sh
    os.chdir("NeMo")
    run_command(["git", "checkout", "master"])
    
    # Check platform and use shell command if bash is required
    if platform.system() == "Windows":
        print("For Windows, ensure you have Git Bash installed to run bash scripts.")
        run_command(["bash", "reinstall.sh"], shell=True)
    else:
        run_command(["bash", "reinstall.sh"])
    
    os.chdir("..")  # Return to the original directory

    # Step 6: Download the checkpoint.nemo file
    download_file("https://huggingface.co/logeshkg/NeMo/resolve/main/checkpoint.nemo?download=true", "checkpoint.nemo")

    print("Done")

if __name__ == "__main__":
    main()
