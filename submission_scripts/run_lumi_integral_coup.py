import os
import numpy as np

if __name__ == '__main__':
    scanType = 'x'  # other possibilities: 'y', 'xy'
    separations =  [0] #np.arange(3,6,1) #np.arange(0,5.5,0.5)
    # print(separations)
    couplings = [0.0,0.0001,0.0003,0.0009,0.001] #np.linspace(1e-4, 1e-3, 10) #[0.0, 0.0001, 0.001]
    # couplings = np.insert(couplings, 0, 0.0)
    # couplings = [0, 5, 8, 16]
    # print(couplings)
    # particles = [1e5,1e6,1e7]
    # cells = [100,300,600,900,1200]
    #print(cells)
    path = '/afs/cern.ch/work/l/llammert/public/xsuite_project'
    template_path = os.path.join(path, 'bbcode_coupscan.py')
    output_directory = os.path.join(path, './scripts/lhc_flat_bb')  # Directory to save modified scripts

    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Read the original template file once and store its content
    with open(template_path, 'r') as file:
        original_text = file.read()

    # Loop through each configuration setting
    # for p in particles:
    #     for c in cells :
    for sep in separations:
        for coup in couplings:
            #print(f"Generating scripts for sep = {sep}, coupling = {c}")
            coup = round(coup,4)
            # Use the original text for each modification to ensure the template remains unchanged
            text = original_text
            text = text.replace('coupling = 0.0', f'coupling = {coup}')
            text = text.replace('ksl = 0.0', f'ksl = {coup}')
            # text = text.replace('n_macroparticles = int(1e6)',f'n_macroparticles = int({p})')
            # text = text.replace('lumi_cells = 300',f'lumi_cells = {c}')
            text = text.replace('xshift = 0.0', f'xshift = {sep}')
            text = text.replace('totalshift = 0.0', f'totalshift = {sep}')  
            

            modified_filename = f"{coup}coup_{sep}sep.py"    #f"{coup}coup_{sep}sep.py"
            modified_file_path = os.path.join(output_directory, modified_filename)

            # Save modified script to a new file
            with open(modified_file_path, 'w') as modified_file:
                modified_file.write(text)
                
            print(f"Submitting job with settings: coup = {coup}, sep = {sep}")
            # Adjust the job submission to use a consistent shell script
            os.system(f'condor_submit run_lumi_integral.sub -a "arguments={modified_file_path}"')
            # os.system(f'python {modified_file_path}')

