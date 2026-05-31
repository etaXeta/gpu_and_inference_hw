import subprocess
import sys
from pathlib import Path

def main():
    hw2_dir = Path(__file__).parent
    results_dir = hw2_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    output_file = results_dir / "hw2_output.txt"
    task_script = hw2_dir / "hw2_task.py"

    print(f"Running {task_script} and capturing output to {output_file}...")

    # We use subprocess to run the task script and capture both stdout and stderr
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            process = subprocess.Popen(
                [sys.executable, str(task_script)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8"
            )
            
            # Read output in real-time and write to file and print to console
            for line in process.stdout:
                print(line, end="")
                f.write(line)
            
            process.wait()
            
            if process.returncode != 0:
                print(f"\nWarning: {task_script} exited with return code {process.returncode}")
            else:
                print(f"\nSuccessfully saved output to {output_file}")
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
