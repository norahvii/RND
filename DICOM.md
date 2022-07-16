# DICOM files  
### DICOM (.dcm)  
### Nifity (.nii)

### Medical imaging files.  
DICOM is a data structure and network protocol.  
Digital Imaging and Communications in Medicine.  

## Data Structure:  
Key - value pair  
http://dicomlookup.com/getpdf2016.asp  
Group ([0-9]{4}, [0-9]{4}), data element  
(tag group, element group)  
```
(0002,0000) UL 210                           # 4,1 File Meta Information Group Length
(0002,0002) UI [1.2.840.10008.5.1.4.1.1.2]   # 26,1 Media Storage SOP Class UID
(0008,1090) LO [NeuViz 16 ]                  # 10,1 Manufacturer's Model Name
```  
Each DICOM file has their own metadata.  
So we must take the pixel data out of each DICOM file and put them into NumPy arrays in the python coding. Pixel data describes a color image with a single sample per pixel. Essentially, NumPy arrays are used for matrix math, deep learning models, and so on. For our purposes, we will require a specific Python package for DICOM metadata editing: Pydicom.  
https://anaconda.org/conda-forge/pydicom  

### How to think about Pixel Data:
1. Field of view (the size in cm of the area presented by an image)
2. Pixel size (the size in mm of the x and y coordinates of a pixel)
3. Slice thickness: (the size in mm of the z coordinate of a pixel)

Voxels are like pixels but with a z coordinate allowing you to combine cubes into 3D shapes. Game developers abandoned using them over the mighty triangle because triangles are easier for graphic hardware to process.  

0 - 255 Black and White Pixel range.  
The darker a pixel is, the closer it is to 0  
The lighter a pixel is, the closer it is to 255  
The more bits we have, the greater the bit range and the higher quality image.  

Radiologists evaluate these pixels and the image by means of a quantitative value called HU.  
## Hounsfield Unites = HU  
"In a CT scan, Hounsfield Unit is proportional to the degree of x-ray attenuation and it is allocated to each pixel to show the image that represents the density of the tissue."

$$HU = \left(\frac{\mu\space{\text{material}}-\mu\space{\text{water}}}{\mu\space{\text{water}}}\right)\times 1000$$  

An HU unit is a conversion of pixel data into an HU value. The attenuation co-efficient is the number left over from the kW that is subtracted from the living material jamming the X-rays. i.e. 100 kW - 80 kW = 20 kW. Means 80 Kw of living material interfeared with the ray's full power. The computer saves 20 kW in binary.     
Hence, Attenuation Co-efficient is 20 kW > Pixel Size (the size of the pixel in millimeters is 0.05 mm) > Hounsfield units: absorbtion rate of X-rays.

Hu value for air is -1000 HU  
Hu value for water is 0 HU  
Hu value for blood air is 35-40 HU  

These values make the range four our construction.  

Y = mx+b  
where "m" is the slope and "b" is the intercept

if X = 77.137497  
m = 1.211  
b = -64.434
Our HU value is 28.979  
HU is a linear transformation of original linear attenuation coefficient measurement.

```py
# Convert to Hounsfield units (HU)
for slice_number in range(len(slices)):
    intercept = slices[slice_number].RescaleIntercept
# DICOM metadata attribute call
    slope = slices[slice_number].RescaleSlope
# DICOM metadata attribute call
    if slope != 1:
        image[slice_number] = slope * image[slice_number].astype(np.float64)
        image[slice_number] = image[slice_number].astype(np.int16)
```

For each slice, we will take the Rescaled Intercept and the Rescaled Slope out of the data and use that to calculate the HU values [HU values inform us which pixels will be a part of whatever tissues].

Different machines will have different bit pixel encoding values. What we've explained thusfar is the data itself, the data structure, what a DICOM file is and what it contains, and preprocessing (converting the data into values that are useable by a deep learning model).

## Supplemental:
* Code section:
https://www.youtube.com/watch?v=hWwAFNmPZFQ  
* Download some DICOMs: https://figshare.com/collections/FUMPE/4107803  
* Homework: https://www.youtube.com/watch?v=KZld-5W99cI&t=0s  
* Full preprocessing tutorial: https://www.kaggle.com/code/gzuidhof/full-preprocessing-tutorial/notebook  
* How DICOM works: https://www.youtube.com/watch?v=eCECXr-HxVs  
* A Python script to sort DICOM files: https://towardsdatascience.com/a-python-script-to-sort-dicom-files-f1623a7f40b8  
* nibabel: https://nipy.org/nibabel/  
* SimpleITK Jupyter Notebooks: https://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/  

### High level overview of DICOM:  
* Medical imaging devices create DICOM files.  
* DICOM files contain the image data, information about the patient (such as name, ID, sex, and DOB), and other information [such as acquisition data: type of equipment used, settings on the modality] about the image.  
* DICOM contains network protocols.  
  - TCP / IP, and HTTP + [ It uses other protocols, such as DHCP, SAMIL ... ]
  - More ntwork commands that are used to schedule procedures, report statuses between doctors and medical devices.
* DICOM identifies something by a collection of it's attributes which is called an information object definition or IOD. 
  - "Information Object Definition" == IOD
  - IOD's are used for many things: Device IOD, Contrast IOD, CT image IOD
* DICOm uses services to perform functions with the objects.  
* A DICOM object plus a DICOM service equals a "Service Object Pair (SOP).  
  - Performing services like "Move", "Find" and "Store"
  - https://dicomlibrary.com/dicom/sop/
* The DICOM networking protocol details a specific procedures for DICOM devices to connect including a DICOM handshake.

### Anatomy of DICOM File:

| Preamble 128 bytes | Prefix 4 bytes | Tag | VR | Length | Value |  
| :--- | :----: | :----: | :----: |:----: | ---: |  

``` 
00000000:  0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000  :........................
00000048:  0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000  :........................
00000060:  0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000  :........................
00000078:  0000 0000 0000 0000 4449 434d 0200 0000 554c 0400 d200 0000  :........DICM....UL......
00000090:  0200 0100 4f42 0000 0200 0000 0001 0200 0200 5549 1a00 312e  :....OB............UI..1.
000000a8:  322e 3834 302e 3130 3030 382e 352e 312e 342e 312e 312e 3200  :2.840.10008.5.1.4.1.1.2.
000000c0:  0200 0300 5549 4000 312e 332e 362e 312e 342e 312e 3935 3930  :....UI@.1.3.6.1.4.1.9590
00000138:  3935 3930 2e31 3030 2e31 2e33 2e31 3030 2e39 2e34 0200 1300  :9590.100.1.3.100.9.4....
00000150:  5348 0e00 4d41 544c 4142 2049 5054 2039 2e34 0800 0500 4353  :SH..MATLAB IPT 9.4....CS
00000168:  0a00 4953 4f5f 4952 2031 3030 0800 0800 4353 1e00 4f52 4947  :..ISO_IR 100....CS..ORIG
```  

### Linux Tools:
* libgdcm-tools  
  - Grassroots DICOM tools and utilities  
```console
sudo apt install libgdcm-tools  
gdcmdump [options] dcm_file  
gdcmdump [options] dcm_directory  
```  
* dcm2niix  
  - next generation DICOM to NIfTI converter
* dicomnifti
  - converts DICOM files into the NIfTI format
...

### Python Tools: 
* dicom2nifti: https://github.com/icometrix/dicom2nifti  
```py
import dicom2nifti

dicom2nifti.convert_directory(dicom_directory, output_folder, compression=True, reorient=True)
```