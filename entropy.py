import file_entropy
from sys import argv, exit

if __name__ == "__main__": 
    
    if len(argv) < 2: 
        print("Error: must have at least 1 argument for file path")
        exit(-1)

    f_path = argv[1]

    print(f"Computing entropy of file '{f_path}'...", end='')

    entropy = file_entropy.shannon(f_path)
    
    print(f"done.\nEntropy: {entropy}")

    exit(0)