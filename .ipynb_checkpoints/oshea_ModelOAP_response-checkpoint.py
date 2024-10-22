## this is from https://github.com/seboshea/OAPmodel/blob/main/ModelOAP_response.py
## STATS for images
# -*- coding: utf-8 -*-

# Model optical array probe response from 2D binary shape at different Z (position between arms)
# Uses angular spectrum theory

#References
#https://amt.copernicus.org/articles/12/2513/2019/
#https://amt.copernicus.org/articles/14/1917/2021/amt-14-1917-2021.html


import numpy as np
import matplotlib.pyplot as plt 
#from skimage.draw import ellipse
from skimage.measure import label, regionprops
#from skimage.transform import rotate
import h5py
import scipy.io
import imageio
import os
from skimage import filters
from scipy import ndimage


#_______________________________________________________________________________________

# Theoretical shadow images of 2D non-spherical shapes are calculated 
# using a diffraction model based on angular spectrum theory. Particle stats 
# (diameter, area fraction, greyscale ratios etc) are output as a function 
# of Z (position between probe arms)

# ShapeFlag set what is used as the initial 2D shape
# ShapeFlag = 0 Generate shape which is hard coded shape in ShapeVsZ().
# ShapeFlag = 1 Use CPI image (.jpg) for image.
# ShapeFlag = 2 Use general binary image 1 = background, 0 = shape. Array needs to be (2048,2048) elements.

# PixelSizeInput = pixel size for diffraction calculations
# PixelSizeOutput = pixel size of oap 
# Only tested for PixelSizeInput = 1 and PixelSizeOutput = 10, but should
# work as long as (PixelSizeOutput / pixel_size needs) is an integer.

#Prefix is for output filename


def ShapeVsZ(SavePath,Prefix,ShapeFlag,SourceImage,PixelSizeInput,PixelSizeOutput,PlotDataFlag):

    SaveDataFlag = 1
    
    Lambda= 0.658 # CIP laser wavelength
    #Lambda= 0.785 # 2DS laser wavelength
    Zarray = np.arange(0, 100000, 1000) # Z position to calculate particle stats

    # Parameters    
    # Set up shape
    if ShapeFlag == 0 : 
        
        pixel_size = PixelSizeInput #um
        x_min = -1024*pixel_size # (µm)
        x_max = 1023*pixel_size # (µm) use 2**n for fast FFT
        y_min = -1024*pixel_size # (µm)
        y_max = 1023*pixel_size # (µm) use 2**n for fast FFT
        x = np.arange(x_min, x_max+.001, pixel_size)
        y = np.arange(y_min, y_max+.001, pixel_size)
        X, Y = np.meshgrid(x, y)
        
        ### Circle
        #D= 150 # um
        #mask = X**2 + Y**2 < (D/2)**2 # Circle
        
        ###Rectangle
        W = 50
        H = 300
        mask = (abs(X) < (W/2)) & (abs(Y) < (H/2)) # rectangle
        M= np.ones((x.size,y.size))
        M[mask] = 0
        pixel_size = 1
    
    if ShapeFlag == 1 : # Use CPI image as mask
        pixel_size = PixelSizeInput #um
        x_min = -1024*pixel_size # (µm)
        x_max = 1023*pixel_size # (µm) use 2**n for fast FFT
        y_min = -1024*pixel_size # (µm)
        y_max = 1023*pixel_size # (µm) use 2**n for fast FFT
        x = np.arange(x_min, x_max+.001, pixel_size)
        y = np.arange(y_min, y_max+.001, pixel_size)
        X, Y = np.meshgrid(x, y)
        M=LoadCpiImageMask(SourceImage)
        #pixel_size = 1
    
    if ShapeFlag == 2 : # Use general binary image as mask
        pixel_size = PixelSizeInput #um
        x_min = -1024*pixel_size # (µm)
        x_max = 1023*pixel_size # (µm) use 2**n for fast FFT
        y_min = -1024*pixel_size # (µm)
        y_max = 1023*pixel_size # (µm) use 2**n for fast FFT
        x = np.arange(x_min, x_max+.001, pixel_size)
        y = np.arange(y_min, y_max+.001, pixel_size)
        X, Y = np.meshgrid(x, y)
        M = SourceImage
        
    
    # These areas and diameter are for all pixels where the drop in light intestiy is greater than threshold.
    # Be aware of this when comparing with OShea AMT 2019. 
    DiameterLevel0_BG=np.zeros(len(Zarray))
    DiameterLevel1_BG=np.zeros(len(Zarray))
    DiameterLevel2_BG=np.zeros(len(Zarray))
    Area0_BG =np.zeros(len(Zarray)) 
    Area1_BG =np.zeros(len(Zarray))
    Area2_BG =np.zeros(len(Zarray))
    Area0_Filled =np.zeros(len(Zarray)) 
    Area1_Filled =np.zeros(len(Zarray))
    Area2_Filled =np.zeros(len(Zarray))
    Circularity0_filled =np.zeros(len(Zarray)) 
    Circularity1_filled =np.zeros(len(Zarray))
    Circularity2_filled =np.zeros(len(Zarray))
    BoxDiameterLevel0=np.zeros(len(Zarray))
    BoxDiameterLevel1=np.zeros(len(Zarray))
    BoxDiameterLevel2=np.zeros(len(Zarray))
    DiameterBGy1=np.zeros(len(Zarray))
    DiameterBGx1=np.zeros(len(Zarray))
    Area_BBoxLevel0=np.zeros(len(Zarray))
    Area_BBoxLevel1=np.zeros(len(Zarray))
    Area_BBoxLevel2=np.zeros(len(Zarray))
    Perimeter_BBox_FilledLevel1 =np.zeros(len(Zarray))
    Area_BBox_FilledLevel1 =np.zeros(len(Zarray))
    MaxD_BBoxLevel1 =np.zeros(len(Zarray))
    MaxD_BGLevel1 =np.zeros(len(Zarray))
    DiameterX=np.zeros(len(Zarray))
    DiameterY=np.zeros(len(Zarray))
    
    #Calculate diffraction for each Z in Zarray
    for i in range(len(Zarray)):
        Z= Zarray[i]
        #print(Z)
        I, A0, fx, fy = compute_diffraction(Z, Lambda, pixel_size, x, y, X, Y, M)
        #average image to OAP resolution and apply greyscale thresholds
        AveragingFactor = PixelSizeOutput / pixel_size # This needs to be an integer
        xOAP, yOAP, I_binnned, I_binned_2, I_binned_1, I_binned_0= AverageFactorOAPpixels(I, x, y, AveragingFactor, PixelSizeOutput)
        
        ImageStatsDict0= ImageParticleMoreStats(I_binned_0, PixelSizeOutput)
        ImageStatsDict1= ImageParticleMoreStats(I_binned_1, PixelSizeOutput)
        ImageStatsDict2= ImageParticleMoreStats(I_binned_2, PixelSizeOutput)
        
        DiameterLevel0_BG[i]=ImageStatsDict0['MeanXY_BG']
        DiameterLevel1_BG[i]=ImageStatsDict1['MeanXY_BG']
        DiameterLevel2_BG[i]=ImageStatsDict2['MeanXY_BG']
        Area0_BG[i]=ImageStatsDict0['Area_BG'] 
        Area1_BG[i]=ImageStatsDict1['Area_BG'] 
        Area2_BG[i]=ImageStatsDict2['Area_BG'] 
        Area0_Filled[i]=ImageStatsDict0['Area_Filled'] 
        Area1_Filled[i]=ImageStatsDict1['Area_Filled']
        Area2_Filled[i]=ImageStatsDict2['Area_Filled']
        Circularity0_filled[i]=ImageStatsDict0['Circularity_Filled']
        Circularity1_filled[i]=ImageStatsDict1['Circularity_Filled']
        Circularity2_filled[i]=ImageStatsDict2['Circularity_Filled']
        BoxDiameterLevel0[i]=ImageStatsDict0['MeanXY']
        BoxDiameterLevel1[i]=ImageStatsDict1['MeanXY']
        BoxDiameterLevel2[i]=ImageStatsDict2['MeanXY']
        DiameterBGy1[i]=ImageStatsDict1['DiameterBGx']
        DiameterBGx1[i]=ImageStatsDict1['DiameterBGx']
        Area_BBoxLevel0[i]=ImageStatsDict0['Area_BBox']
        Area_BBoxLevel1[i]=ImageStatsDict1['Area_BBox']
        Area_BBoxLevel2[i]=ImageStatsDict2['Area_BBox']
        Perimeter_BBox_FilledLevel1[i]=ImageStatsDict1['Perimeter_BBox_Filled']
        Area_BBox_FilledLevel1[i]=ImageStatsDict1['Area_BBox_Filled']
        MaxD_BBoxLevel1[i]=ImageStatsDict1['DmaxBBox']
        MaxD_BGLevel1[i]=ImageStatsDict1['DmaxBG']
        DiameterX[i]=ImageStatsDict1['DiameterX']
        DiameterY[i]=ImageStatsDict1['DiameterY']
        
        D0 = DiameterLevel1_BG[0]
        if PlotDataFlag : 
            Figurename=SavePath+Prefix+'at'+str(Z)+'.png.'
            Zd_true = (4 * Lambda * Z) / ((D0)**2) 
            plot_diffraction(xOAP, yOAP, I_binned_0, I_binned_1,I_binned_2,x,y, M,I,A0, fx, fy, Z,Zd_true, 1, Figurename)

                       
    #greyscale ratios
    AreaFraction0=(Area0_BG-Area1_BG)/Area0_BG # Alow notation from OShea AMT 2019   
    AreaFraction1=(Area1_BG-Area2_BG)/Area0_BG # Amid
    AreaFraction2=Area2_BG/Area0_BG # Ahigh   
    
    #Save image stats
    if SaveDataFlag == 1 : 
        FileName= Prefix+'.h5'
        try:
            os.remove(SavePath+FileName)
        except OSError:
            pass   
        h5f = h5py.File(SavePath+FileName, 'w')
    
        h5f.create_dataset('Zarray', data=Zarray)
        h5f.create_dataset('DiameterLevel0', data=DiameterLevel0_BG)
        h5f.create_dataset('DiameterLevel1', data=DiameterLevel1_BG)
        h5f.create_dataset('DiameterLevel2', data=DiameterLevel2_BG)
        h5f.create_dataset('AreaFraction0', data=AreaFraction0)
        h5f.create_dataset('AreaFraction1', data=AreaFraction1)
        h5f.create_dataset('AreaFraction2', data=AreaFraction2)
        h5f.create_dataset('Circularity0_filled', data=Circularity0_filled)
        h5f.create_dataset('Circularity1_filled', data=Circularity1_filled)
        h5f.create_dataset('Circularity2_filled', data=Circularity2_filled)
        h5f.create_dataset('BoxDiameterLevel0',data=BoxDiameterLevel0)
        h5f.create_dataset('BoxDiameterLevel1',data=BoxDiameterLevel1)
        h5f.create_dataset('BoxDiameterLevel2',data=BoxDiameterLevel2)
        h5f.create_dataset('Area0_Filled',data=Area0_Filled)
        h5f.create_dataset('Area1_Filled',data=Area1_Filled)
        h5f.create_dataset('Area2_Filled',data=Area2_Filled)
        h5f.create_dataset('Area0_BG',data=Area0_BG)
        h5f.create_dataset('Area1_BG',data=Area1_BG)
        h5f.create_dataset('Area2_BG',data=Area2_BG)
        h5f.create_dataset('DiameterBGx1',data=DiameterBGx1)
        h5f.create_dataset('DiameterBGy1',data=DiameterBGy1)
        h5f.create_dataset('Area_BBoxLevel0',data=Area_BBoxLevel0)
        h5f.create_dataset('Area_BBoxLevel1',data=Area_BBoxLevel1)
        h5f.create_dataset('Area_BBoxLevel2',data=Area_BBoxLevel2)
        h5f.create_dataset('Perimeter_BBox_FilledLevel1',data=Perimeter_BBox_FilledLevel1)
        h5f.create_dataset('Area_BBox_FilledLevel1',data=Area_BBox_FilledLevel1)
        h5f.create_dataset('MaxD_BBoxLevel1 ',data=MaxD_BBoxLevel1 )
        h5f.create_dataset('MaxD_BGLevel1',data=MaxD_BGLevel1)
        
        h5f.create_dataset('DiameterX',data=DiameterX)
        h5f.create_dataset('DiameterY',data=DiameterY)
      
        h5f.close()



#_______________________________________________________________________________________

#Average AveragingFactor elements of 2D array ssuming that each dimension of the new shape is a factor of the corresponding dimension in the old one


def AverageFactorOAPpixels(I, x, y, AveragingFactor, OAP_PixelSize) : 

    I_subset=I[int(I.shape[0]/2)-1000: int(I.shape[0]/2) + 1000, int(I.shape[1]/2)-1000: int(I.shape[1]/2) + 1000]
   
    nsmallx = int(I_subset.shape[0] / AveragingFactor)
    nsmally = int(I_subset.shape[1] / AveragingFactor)
    AveragingFactor =int(AveragingFactor)
    
    # Average to 2DS size bins
    I_binnned = I_subset.reshape([nsmallx, AveragingFactor, nsmally, AveragingFactor]).mean(3).mean(1)

    x_bins = np.arange(x[int(I.shape[0]/2)-1000], x[int(I.shape[0]/2) + 1000], OAP_PixelSize)
    y_bins = np.arange(x[int(I.shape[1]/2)-1000], x[int(I.shape[1]/2) + 1000], OAP_PixelSize)
    
    # 25, 50 , 75 intensity thresholds
    I_binned_2 = (np.where(I_binnned<0.25, 1, 0))
    I_binned_1 = (np.where(I_binnned<0.5, 1, 0))
    I_binned_0 = (np.where(I_binnned<0.75, 1, 0))

    # 25, 50 , 65 intensity thrsholds 
    #I_binned_2 = (np.where(I_binnned<0.35, 1, 0))
    #I_binned_1 = (np.where(I_binnned<0.5, 1, 0))
    #I_binned_0 = (np.where(I_binnned<0.75, 1, 0))
    
    # 25, 50 , 85 intensity thrsholds 
    #I_binned_2 = (np.where(I_binnned<0.15, 1, 0))
    #I_binned_1 = (np.where(I_binnned<0.5, 1, 0))
    #I_binned_0 = (np.where(I_binnned<0.75, 1, 0))
    
    # 25, 50 , 70 intensity thrsholds 
    #I_binned_2 = (np.where(I_binnned<0.3, 1, 0))
    #I_binned_1 = (np.where(I_binnned<0.5, 1, 0))
    #I_binned_0 = (np.where(I_binnned<0.75, 1, 0))
    
    # 25, 50 , 80 intensity thrsholds 
    #I_binned_2 = (np.where(I_binnned<0.2, 1, 0))
    #I_binned_1 = (np.where(I_binnned<0.5, 1, 0))
    #I_binned_0 = (np.where(I_binnned<0.75, 1, 0))
    
    return x_bins,y_bins, I_binnned, I_binned_2, I_binned_1, I_binned_0


#_______________________________________________________________________________________

# Compute diffraction intensity (I) at a given Z

# from #https://amt.copernicus.org/articles/12/2513/2019/

def compute_diffraction(Z, l, pixel_size, x, y, X, Y, M):

    # Parameters
    k = 2*np.pi/l
    
    delta_fx = 1./(x.size*pixel_size) # In order to be able to use FFT, Eq. (4.4-12) of Ersoy (2006)
    delta_fy = 1./(y.size*pixel_size)
    fx_max = x.size/2*delta_fx # µm-1
    fy_max = y.size/2*delta_fy # µm-1
    fx = np.arange(-fx_max, fx_max-delta_fx+.0000001, delta_fx)
    fy = np.arange(-fy_max, fy_max-delta_fy+.0000001, delta_fy)

    FX, FY = np.meshgrid(fx, fy)

    # Field at Z = 0
    U0 = np.copy(M)

    # Angular spectrum at Z = 0 (Fourier tranform), Eq. (4.3-2) of Ersoy (2006) 
    A0 = np.fft.fftshift(np.fft.fft2(U0))
    # fft2: Compute the 2-dimensional discrete Fourier Transform
    # fftshift: Shift the zero-frequency component to the center of the spectrum

    # Transfer function, Eq. (4.3-14) of Ersoy (2006)
    non_homogeneous_waves = 4*np.pi**2*(FX**2+FY**2) > k**2 # Eq. (4.3-13) of Ersoy (2006)
    H = np.exp( 1j*Z*np.abs(np.sqrt(k**2-4*np.pi**2*(FX**2+FY**2))) )
    H[non_homogeneous_waves] = 0

    # Angular spectrum at Z = Z, Eq. (4.3-11) of Ersoy (2006)
    AZ = A0*H

    # Field at Z = Z, Eq. (4.3-12) of Ersoy (2006)
    UZ = np.fft.ifft2(np.fft.fftshift(AZ))

    # Intensity at Z = Z
    I = np.abs(UZ)**2

    # Phase at Z = Z
    #P = np.arctan(np.imag(UZ)/np.real(UZ))
    
    return I, A0, fx, fy 


#_______________________________________________________________________________________

# When using CPI image for initial particle shape
  
def LoadCpiImageMask(ImagePath):
    
    MaskArray = np.ones((2048,2048)) 
    import_image = imageio.imread(ImagePath,as_gray=True)
    val = filters.threshold_otsu(import_image)
    BinaryImage= (np.where(import_image<val, 0, 1))
    shape=np.shape(BinaryImage)
    xlen=shape[0]
    ylen=shape[1]
    
    MaskArray[1024-int(xlen/2) : 1024+int(xlen/2+0.5), 1024-int(ylen/2+0.5) :1024+int(ylen/2) ]= BinaryImage
                    
    return MaskArray

#_______________________________________________________________________________________

#Calculate statistics of image

#Get stats for biggest paricle in region used example from https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html#sphx-glr-auto-examples-segmentation-plot-label-py


def ImageParticleMoreStats(BinaryImage, OAPPixelSize):
    
    ImageStatsDict = {}
    
    if np.sum(BinaryImage) > 0 :
        #all particles in image
        BoxStats = regionprops(BinaryImage, cache=False)
        Boxbbox = [r.bbox for r in BoxStats]
        ImageStatsDict['MeanXY'] = OAPPixelSize * (Boxbbox[0][2] - Boxbbox[0][0] + Boxbbox[0][3] - Boxbbox[0][1]) / 2
        ImageStatsDict['Area_BBox'] = BoxStats[0].area
        ImageStatsDict['DmaxBBox']  = OAPPixelSize * BoxStats[0].major_axis_length
        ImageStatsDict['DiameterX'] = OAPPixelSize * (Boxbbox[0][3] - Boxbbox[0][1])
        ImageStatsDict['DiameterY'] = OAPPixelSize * (Boxbbox[0][2] - Boxbbox[0][0])
        
        #fill internal voids
        FilledImage = ndimage.morphology.binary_fill_holes(BinaryImage).astype(int)
        FilledStats = regionprops(FilledImage, cache=False)
        ImageStatsDict['Perimeter_BBox_Filled'] = FilledStats[0].perimeter
        ImageStatsDict['Area_BBox_Filled'] = FilledStats[0].area

        #select largest particle in images
        labels_max= SelectLargestParticle(BinaryImage)
        stats = regionprops(labels_max, cache=False)
        bbox = [r.bbox for r in stats]                     
        ImageStatsDict['MeanXY_BG'] = OAPPixelSize * (bbox[0][2] - bbox[0][0] + bbox[0][3] - bbox[0][1]) / 2
        ImageStatsDict['DiameterBGx'] = OAPPixelSize * (bbox[0][3] - bbox[0][1])
        ImageStatsDict['DiameterBGy'] = OAPPixelSize * (bbox[0][2] - bbox[0][0])
        #Area_BG= [r.area for r in stats]
        #Area_BG=Area_BG[0] # pixels     
        ImageStatsDict['Area_BG'] = stats[0].area
        ImageStatsDict['DmaxBG']  = OAPPixelSize * BoxStats[0].major_axis_length
       
        
        #fill internal voids largest particle in image
        FilledImageBG = ndimage.morphology.binary_fill_holes(labels_max).astype(int) # fill internal voids for circularity 
        FilledStatsBG = regionprops(FilledImageBG, cache=False) 
        ImageStatsDict['Circularity_Filled'] = FilledStatsBG[0].perimeter** 2 / (4 * np.pi * FilledStatsBG[0].area )
        #Area_Filled = [r.area for r in stats]
        ImageStatsDict['Area_Filled'] = FilledStatsBG[0].area
    else:
        
        ImageStatsDict['MeanXY_BG'] =0
        ImageStatsDict['Area_BG'] =0
        ImageStatsDict['DiameterBGx'] =0
        ImageStatsDict['DiameterBGy'] =0
        ImageStatsDict['Circularity_Filled'] =np.nan
        ImageStatsDict['MeanXY'] =0
        ImageStatsDict['Area_Filled'] = 0
        ImageStatsDict['Area_BBox'] = 0
        ImageStatsDict['Perimeter_BBox_Filled'] =0
        ImageStatsDict['Area_BBox_Filled']=0
        ImageStatsDict['DmaxBBox']= 0 
        ImageStatsDict['DmaxBG'] = 0 
        ImageStatsDict['DiameterX'] = 0
        ImageStatsDict['DiameterY'] = 0 
    return ImageStatsDict


#_______________________________________________________________________________________

# Select largest particle in image

def SelectLargestParticle(segmentation):
    labels = label(segmentation)
    unique, counts = np.unique(labels, return_counts=True)
    list_seg=list(zip(unique, counts))[1:] # the 0 label is by default background so take the rest
    largest=max(list_seg, key=lambda x:x[1])[0]
    labels_max=(labels == largest).astype(int)
    return labels_max

#_______________________________________________________________________________________

# Plot calculated image 

def plot_diffraction(xOAP,yOAP, ImageLow, ImageMid, ImageHigh,x, y, imageZ0, imageDiffraction,A0, fx, fy, Z, Zd,SaveFlag, Figurename):
    
    imageGreyscale = (ImageLow - ImageMid )*25 + (ImageMid - ImageHigh )*50 + (ImageHigh )*75
      
    fig=plt.figure(figsize=(8,8)) 
    plt.rcParams.update({'font.size': 12})
    plt.title('z='+str(Z/1000) + ' mm, Zd = '+str(np.around(Zd,2)))
    plt3= plt.pcolormesh(xOAP,yOAP,imageGreyscale, cmap='gist_stern_r')
    plt3.set_clim(vmin=0, vmax=75)
    plt.ylabel('y, μm')
    plt.xlabel('x, μm')
    plt.ylim([-200,200])
    plt.xlim([-200,200])
    
    if SaveFlag == 1 :
        #SavePath = 'C:/Users/Admin TEMP/Documents/Diffraction/Plots/'
        plt.savefig(Figurename,dpi=200)
        plt.close(fig)

#_______________________________________________________________________________________

SavePath = '/home/seb/Documents/OAP_model/'
Prefix = 'Column'
ShapeFlag = 0
SourceImage = ''
PixelSizeInput = 1
PixelSizeOutput = 10

ShapeVsZ(SavePath,Prefix,ShapeFlag,SourceImage,PixelSizeInput,PixelSizeOutput,True)
