# Content of utilities.py
# For example, a simplified version or the full one if fetched
import os
import sys
import platform
import subprocess

def custom_install_cmdstan(version="2.36.0", cores=2, progress=True, overwrite=False, verbose=False):
    """
    Install CmdStan to a temporary Colab location.
    This is a simplified version for illustration.
    The actual script from the URL is more robust.
    """
    print(f"Attempting to install CmdStan version {version}...")
    cmdstan_path = os.path.join("/tmp", f"cmdstan-{version}")

    if os.path.exists(cmdstan_path) and not overwrite:
        print(f"CmdStan path {cmdstan_path} already exists. Set overwrite=True to reinstall.")
        os.environ["CMDSTAN"] = cmdstan_path
        print("CmdStan path set.")
        return

    try:
        print("Downloading CmdStan...")
        url = f"https://github.com/stan-dev/cmdstan/releases/download/v{version}/cmdstan-{version}.tar.gz"
        subprocess.run(["curl", "-L", "-o", f"cmdstan-{version}.tar.gz", url], check=True)
        
        print("Unpacking CmdStan...")
        subprocess.run(["tar", "-xzf", f"cmdstan-{version}.tar.gz", "-C", "/tmp"], check=True)
        
        print("Building CmdStan...")
        make_cmd = ["make", f"-j{cores}", "build"]
        if platform.system() == "Linux" and "tbb" in subprocess.check_output(["ldd", "--version"]).decode():
             # On systems with TBB, like Colab, specify it
            make_cmd.extend(["TBB_CXX_TYPE_LENGTH_NAME=1"])


        process = subprocess.Popen(make_cmd, cwd=cmdstan_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        if progress:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip(), flush=True)
        else:
            process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"CmdStan build failed with return code {process.returncode}")

        os.environ["CMDSTAN"] = cmdstan_path
        print("CmdStan successfully installed and path set!")

    except Exception as e:
        print(f"An error occurred during CmdStan installation: {e}")
        sys.exit(1)

def test_cmdstan_installation():
    cmdstan_path = os.environ.get("CMDSTAN")
    if not cmdstan_path:
        print("CMDSTAN environment variable not set.")
        return False
    
    model_path = os.path.join(cmdstan_path, "examples/bernoulli/bernoulli.stan")
    data_path = os.path.join(cmdstan_path, "examples/bernoulli/bernoulli.data.json")
    exe_path = os.path.join(cmdstan_path, "examples/bernoulli/bernoulli") # .exe on Windows

    if platform.system() == "Windows":
        exe_path += ".exe"
        
    try:
        print(f"Compiling example model: {model_path}")
        # Clean first
        subprocess.run(["make", "clean-all"], cwd=os.path.join(cmdstan_path, "examples/bernoulli"), check=True, capture_output=True)
        subprocess.run(["make", os.path.join(cmdstan_path, "examples/bernoulli/bernoulli")], cwd=cmdstan_path, check=True, capture_output=True)
        
        print(f"Running example model: {exe_path}")
        cmd = [exe_path, "sample", "data", f"file={data_path}", "output", "file=output.csv"]
        subprocess.run(cmd, check=True, capture_output=True, cwd=".") # Run in current dir to write output.csv here
        
        print("CmdStan installation test passed.")
        if os.path.exists("output.csv"): os.remove("output.csv")
        return True
    except Exception as e:
        print(f"CmdStan installation test failed: {e}")
        if hasattr(e, 'stdout') and e.stdout: print("STDOUT:", e.stdout.decode())
        if hasattr(e, 'stderr') and e.stderr: print("STDERR:", e.stderr.decode())
        return False

if __name__ == '__main__':
    # This part allows running the script directly for installation
    # In a Colab-like environment or where 'curl' and 'tar' are available
    custom_install_cmdstan(overwrite=True, progress=True) # Set overwrite as needed
    if test_cmdstan_installation():
        print("CmdStan is ready.")
    else:
        print("CmdStan installation or test failed. Please check the output.")
