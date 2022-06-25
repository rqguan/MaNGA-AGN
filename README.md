# MaNGA-AGB

# -- Research Journal -- 

Here's the goal of this project:

（1）input the full sample ([OIII] EQW maps in our case), and the code will help to classify the galaxies into different classes.
（2）training the code with bicone galaxies, then tell the code to find similar ones.


## Part One: Bicone Feature Selection
  
  Proposed Prosedure:
  
    1. Transform the galaxy image into a polar coordinate.
    
    2. Stack the galaxy into one histogram.
    
    3. Eliminate any data outlier and smoothening the curve.
    
        a. After that I tried to use earth-moving distance to directly compare the curves but failed. 
        b. It turned out that moving a similar but out-of-phase curve is the same a moving a really noisy curve
        
    4. Fourier Transform the curve and find the strongest frequency.
    
        a. The bicone structure occures twice a cycle. 
        b. With 2 cycles of daya in a curve, the strongest frenquency should be 4 Hz (index 4 in data), assuming the bicone is strong.   [1st Filter]
        
            i. Origin value of the curve need to be set to 0 to avoid saturates the FT @ x=0
            ii. Therefore normalization and minimum value aligned to the origin are required
            
        c. Frequenct above 50Hz makes no sense so they are truncated. 
        d. The difference between the first 4 strongest frequency shall be large, as the sample suggest.                                 [2nd Filter]
        e. Frequency of 1 Hz and 5 Hz are allowed but has to be weaker than 2 Hz, we use the sample to give an empiracle parameter.      [3rd Filter]
        f. During thr process, spacial features are integrated into a radial curve, therefore lost. 
        
    5. Address to the lost of spacial data.
    
        a. Try to do a FT to the r=0R~0.5R as well as r=0.5R~R.
            i. Thus we can eliminate the ones only has two outer bright dots instead of "bicone"
            ii. But dubiously bicones would be washed out. 
    
    
