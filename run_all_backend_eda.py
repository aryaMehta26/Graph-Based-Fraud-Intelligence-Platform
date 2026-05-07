import os
import subprocess
import time

def run_script(script_name):
    print("\n" + "="*80)
    print(f"🚀 EXECUTING {script_name} ON FULL 31.9M DATASET...")
    print("="*80)
    
    start_time = time.time()
    try:
        # Run unbuffered so output streams directly to standard out
        process = subprocess.Popen(["python3", "-u", script_name])
        process.wait()
        
        if process.returncode != 0:
            print(f"\n❌ ERROR: {script_name} failed with exit code {process.returncode}")
            return False
            
        elapsed = time.time() - start_time
        print("\n" + "="*80)
        print(f"✅ {script_name} COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n❌ EXCEPTION executing {script_name}: {e}")
        return False

def main():
    print("""
    ======================================================================
    ENTERPRISE DATA PIPELINE: MASTER EXECUTION BATCH
    Team 12 | DATA 298A | Fraud Intelligence Platform
    ======================================================================
    This master script sequentially enforces data extraction, 
    strict constraint cleaning, and exploratory data visualization 
    generation over the entire 31.9M raw Kaggle database.
    ======================================================================
    """)
    
    # Ensuring we are executing from the correct path relative to the files
    proj_dir = "/Users/aryaaa/Desktop/DATA 298"
    os.chdir(proj_dir)
    
    scripts = [
        "notebooks/01_data_extraction.py",
        "notebooks/02_data_cleaning.py",
        "notebooks/03_eda_visualizations.py"
    ]
    
    total_start = time.time()
    
    for s in scripts:
        if not run_script(s):
            print("\n❌ PIPELINE HALTED DUE TO DOWNSTREAM FAILURE.")
            return
            
    total_elapsed = time.time() - total_start
    print(f"\n🏆 ENTIRE PIPELINE COMPLETE. Total Master Execution Time: {total_elapsed/60:.2f} Minutes")

if __name__ == "__main__":
    main()
