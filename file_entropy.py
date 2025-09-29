import math 
import numpy as np
from collections import Counter 

def shannon(f_path): 
    '''
    Calculate the Shannon entropy of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        float: Entropy value in bits
    '''
    try:
        # open the file in bianary mode
        with open(f_path, 'rb') as f: 
            data = f.read() 

        # check if the file is empty
        if len(data) == 0: return 0.0 

        # get the counts of each type of byte
        byte_counts = Counter(data)

        entropy = 0. 
        data_len = len(data)

        for count in byte_counts.values(): 
            # calculate the shannon entropy of this particular byte
            p = count / data_len

            entropy += -p * np.log2(p)
        
        return entropy 

    # handle exceptions 
    except FileNotFoundError: 
        print(f"Error opening file '{f_path}': {FileNotFoundError}")
        return None, None

    # generic exception    
    except Exception as err: 
        print(f"Something went wrong, {type(err)}: {err}")
        return None, None

