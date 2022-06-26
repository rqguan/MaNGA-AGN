# MaNGA-AGB

# -- Research Journal -- 

Here's the goal of this project:


    1. input the full sample ([OIII] EQW maps in our case), and the code will help to classify the galaxies into different classes.

    2. training the code with bicone galaxies, then tell the code to find similar ones.
    
## General Properties of this pipeline

    1. The complete MaNGA dataset containing 10782 MAPS file is ~250GB.
    2. My Macbook's hard drive is 500GB, with only 300GB left, so I am using SDSS-MARVIN server throught the process. 
    3. 44 galaxies' MAPS file are missing in the dataset: MARVIN response 404 for both downloading and loading.
    4. Some MAPS files contains arrays of 0s, the ifu is pointing nowhere. 
    5. The problems above are well-addressed in the pipeline. 


## Part One: Bicone Feature Selection
  
  Proposed Prosedure:
  
    1. Transform the galaxy image from polar coordinate to Cartesian coordinate.
        
        a. Set galaxy center as the polar origin.
    
    2. Integrate OIII EW flux along R direction, which is Y direction on the Cartesian coordinat, into an array.
    
        a. The array show as a curve, showing the integrated EW flux along R diredction. 
        b. 3-sigma outliers need to be eliminated before the intergration because some values are too absurd. 
    
    3. Smoothening the curve.
    
        a. Use Gaussian Smoothening, set the smoothness to 10, meaning x = average of 10 near neighbors.
            i. 5 each on left and right.
        b. After this step, I tried to use earth-moving distance to directly compare the curves but failed. 
            i. Meant to compare between curves, taking the mean of the 17 samples then compare to each in the dataset. 
        c. It turned out that moving a similar but out-of-phase curve is similar to moving a really noisy curve
        
    4. Fourier Transform the curve and find the strongest frequency.
    
        a. The bicone structure occures twice a cycle. 
        b. With 2 cycles of data in a curve, the strongest frenquency should be 2 Hz (index 4 in data), assuming the bicone is strong.   
        ---- [1st Filter] ----
        
            i. Origin value of the curve need to be set to 0 to avoid saturates the FT @ x=0
            ii. Therefore normalization and minimum value aligned to the origin are required
            
        c. Frequency above 50Hz makes no sense so they are truncated. 
        d. The difference between the first 4 strongest frequency shall be large, as the 17 samples suggest.                                
        ---- [2nd Filter] ----
        e. Frequency of 1 Hz and 3 Hz are allowed but has to be weaker than the 2 Hz peak, we use the sample to limit to an empiracle value.     
        ---- [3rd Filter] ----
        f. During thr process, spacial features are integrated into a radial curve, therefore lost. 
        
    5. Addressing to the lost of spacial data.
    
        a. Try to do a FT to the r=0R~0.5R as well as r=0.5R~R.
            i. Thus we can eliminate the ones only have two outer bright dots instead of a continuous "bicone" though out the R direction.
            ii. Downside is that not-so-obvious bicones would be washed out. 
            iii. Turned out the method does not differentiate between bicone and ellipticity. 
            iv. Try to extract the elliptical edge instead
        b. Integrate the flux along the ellipse of the ellipticity and position angle.
        
    
    
