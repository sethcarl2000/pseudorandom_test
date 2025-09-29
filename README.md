# Tests of pseudorandomness 
Contents: 
### entropy.py 

Uses the fcn defined in **file_entropy.py**, takes the first arg as the file path and computes the entropy = [0, 8). 
As an example, the file **test_rand.txt**, from /dev/random, gives an entropy of ~ 8. the file **not_rand.txt**, which is just a single line of ~1000 0's, gives 0. 

Another example, I coped a cpp file from another project and named it 'large_code_file.txt', and used gzip to make a compressed version (large_code_zip.txt.gz). the entropy of the code file is ~4.5, and the zipped version is ~7.9. 

### lpr.py, exp.py
The lpr.py uses a inear congruent pseudo random number generator, with modulo and multiplier you can see in the file. it also computes the mean, stddev, and moments 1-4. **exp.py** uses this LCRG to make an exponential dist, and plots it. 

### norm2.py 
This takes uniformly-distributed numbers from the **numpy.random.uniform** fcn, and uses the **Box-Muller Transform** to make two normally distribued gaussian varabiles. 
Then, another fcn called 'generate_normal_correlation' takes these two, uncorrelated gaussians and re-comptues two new, gaussian variables (x,y) with a given sigma_x, sigma_y, and corrleation coeff. 

The results are plotted for 1e6 points, and correlations of 0.5, 1., -1., and 0. 
