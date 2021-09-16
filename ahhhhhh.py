# Import the necessary packages
import numpy as np
import pandas as pd
from pylab import figure, cm
import matplotlib
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from astropy.io import fits
from tqdm import tqdm
from astropy.visualization import (MinMaxInterval, AsinhStretch, ImageNormalize)
import glob
import astropy.units as u
import math
from astropy.visualization.interval import ManualInterval
import astropy.convolution as conv
import os
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve

def mockup_extract(dataframe, mockup_number):
  '''
  Input: Pandas dataframe, mockup number as an integer
  Output: Four lists: useable names, accurate names, telescope, and parallax value
  Description: Extracts and returns a dictionary of four columns of information from a given dataframe
  '''

  # Create a dictionary of dataframe column names with simple references
  col_names = {}
  col_names['names'] = 'Mockup '+str(mockup_number)
  col_names['file names'] = 'Mockup '+str(mockup_number)+' File Names'
  col_names['telescope'] = 'Mockup '+str(mockup_number)+ ' Telescope'
  col_names['parallax'] = 'Mockup '+str(mockup_number)+' Parallax'
  col_names['mask_rad'] = 'Mockup '+str(mockup_number)+' Mask Radius'
  col_names['crop'] = 'Mockup '+str(mockup_number)+' Crop'
  col_names['band'] = 'Mockup '+str(mockup_number)+' Band'

  # Initialize a dicitonary to return
  columns = {}
  # For each of the four columns of data
  for column in ['names','file names','telescope','parallax','mask_rad','crop','band']:
    # Create a new dictionary entry of the corresponding dataframe column as a list
    if mockup_number < 6:
      columns[column] = dataframe[col_names[column]].values.tolist()[0:21]
    elif mockup_number <10:
      columns[column] = dataframe[col_names[column]].values.tolist()[0:28]
    else:
      columns[column] = dataframe[col_names[column]].values.tolist()[0:24]
  # Add mockup number
  columns['number'] = mockup_number

  return columns

def mlist_extract(dataframe, total_mockup_number):
  mockups = {}  

  for mockup_number in range(1,total_mockup_number):
    mockup = mockup_extract(dataframe,mockup_number)
    # Add to return dictionary
    mockups['Mockup '+str(mockup_number)] = mockup
  
  mockups['number'] = total_mockup_number
  return mockups

def filename_extract(mockups):
  mnumber = mockups['number']
  file_name_list = []
  for mockup_number in range(1,mnumber):
    fnames = mockups['Mockup '+str(mockup_number)]['file names']
    file_name_list.append(fnames)
  
  file_names = []
  for mockup_fnames in file_name_list:
    for name in mockup_fnames:
      if name not in file_names:
        file_names.append(name)
  return file_names

def get_image_paths(folder_name):
  '''
  Input: Folder name in a specific, pre-defined parent directory, and a bool value to create a new file tree (folder)
  Output: A list of direct paths to all fits files in the input folder and possibly a new local folder
  Description: Returns the names of all fits files in a specific input folder and can create a new folder if instructed
  '''
  # Import? Probably a bad syntactical idea
  import glob
  # Define immutable parent directory
  parent_dir = '/content/drive/MyDrive/Follette Lab/Summer 2021/Projects/Transitional Disk Database/PPVII Images/DATA/'
  # Concat parent directory with the input folder name
  download_path = parent_dir+folder_name
  # Use glob package to search for all .fits files in the given folder and add them to a list
  images = glob.glob(download_path+'/*.fits')

  # Return the list of fits file paths
  return(images)

def make_dir(parent_dir, new_folder_name):
  # Define a new subfolder tree for the new data
    path = os.path.join(parent_dir, new_folder_name)
    # Instruct the local operating system to make the defined file tree
    os.mkdir(path)

def hdul_norm(image,slice_dict):
  '''
  Input: String of .fits file path
  Output: Normalized fits object with header that has the extreme values cut
  Description: Takes a .fits file path and returns a normalized and trimmed fits data file with an empty header
  '''
  # Open input fits image
  hdul = fits.open(image)
  # Extract data from the .fits file
  data = hdul[0].data
  
  # Take a specific slice of the datacube
  for telescope in ['VLT','GEMINI','SUBARU']:
    if telescope in image:
      dat = data[slice_dict[telescope]]
      break
  # Check if there is a special case for the disk
  for name in slice_dict['Special Names']:
    if name in image:
      dat = data[slice_dict['Special Slices'][name]]
      break

  # Close the header file
  hdul.close()

  # Create a new fits header file
  hdul_new = fits.HDUList()

  # Copy the fits image data
  image_cut = dat.copy()

  # Set 0.03 percentile of the image to 0
  image_cut -= np.nanpercentile(image_cut,0.03)
  # Normalize the image to the 99.975th percentile
  norm_dat = image_cut/np.nanpercentile(image_cut,99.975)

  # Append the new image to the newly created fits header object
  hdul_new.append(fits.ImageHDU(data=norm_dat))

  # Return the new fits header object
  return(hdul_new)

def rec_add(string, arr, i=0):
  '''
  Input: A string and array of strings
  Output: A combined string of all the strings in the array added to the first input, separated by forward slashes
  Description: Combine a string and array of strings (file tree) into a singular filepath
  '''
  # Run as long as the string array has more elements
  if i < len(arr):
    # Return the previous input with the next element of the string
    return(rec_add(string + '/' + arr[i],arr,i+1))
  # If there are no more elements to add, return the input string
  else: return string

def insert_path(path, new_folder, position):
  '''
  Input: A filepath as a string, a folder as a string, and an integer that specifies where to insert the new folder into the filepath
  Output: The newly created filepath
  Description: Insert a new folder into a specific place in a filepath
  '''
  # Split the new path into a list of folders (i.e. the file tree)
  new_path = path.split('/')
  # Remove the original_data folder
  new_path.remove('Original_Data')
  for tlsc_folder in ['VLT','GEMINI','SUBARU']:
    if tlsc_folder in new_path:
      new_path.remove(tlsc_folder)
  # Insert a new folder into the file tree
  new_path = np.insert(new_path,position,new_folder)
  # Reassemble the filepath by recursively adding all the elements in the list
  new_path = rec_add(new_path[0],new_path[1:])

  # Return the new path
  return(new_path)

def write_fits_folder(image_arr,slice_dict):
  '''
  Input: A list of .fits file paths
  Output: Returns nothing, writes an output of fully processed .fits files to a specified filepath
  Description: Writes the normalized and scaled .fits files to a folder in google drive
  '''
  # Loop through the images in the input list (and print a progress bar)
  for image in tqdm(image_arr):
    duplicate = False
    # Check to see if this image has already been processed
    # Get the image paths of the already processed images
    processed_images = get_image_paths('Processed Images')
    for processed_path in processed_images:  
      if image.split('/')[-1] in processed_path:
        duplicate = True
    if not duplicate:
      # Initialize a normalized, extreme-cut fits header object
      hdul = hdul_norm(image,slice_dict)
      # Create a new string path for the new .fits file
      write_path = insert_path(image,'Processed Images',-1)
      # Write the new .fits file data into a file
      hdul.writeto(write_path)

def get_fits_name(name_string):
  '''
  Input: File path ending in .fits
  Output: Local file name with .fits removed
  Description: Returns filename from a file path
  '''
  # Split the path into an array of folder and file names and select the last element
  full_file_name = name_string.split('/')[-1]
  # Split the file name from its extension and take the file name
  file_name = full_file_name.split('.')[0]
  # Return the file name
  return(file_name)

def get_images(image_paths):
  '''
  Input: A list of image file paths
  Output: A dictionary with mockup names as keys and fits image data as values
  Description: Imports fits images and returns them in a dictionary
  '''
  images = {}
  for image in tqdm(image_paths):
    hdul = fits.open(image)
    data = hdul[0].data
    images[get_fits_name(image)] = data
    hdul.close()
  return images

def create_figax_strings(n=5,m=5):
  '''
  Input: The dimensions of the figure (aka the number of subplots)
  Output: Two strings, one to define the figure and axis, and the other to define an axes variable
  Description: Creates two executable strings than can be run to create a multipanel subplot with the specified dimensions
  '''
  # Initialize main string which will contain all the categorized axes
  main_str = '['
  # Enter first loop, iterating through what will be figure rows
  for i in range(n):
    # Initialize temporary axis string
    ax = '['
    # Enter second loop, iterating through what will be elements in each row
    for j in range(m):
      # If this ax is the first in a row, we don't need a comma
      if ax[-1] == '[':
        ax = ax+'ax'+str(i)+str(j)
      # If this ax isn't the first in a row, we need a comma
      else:
        ax = ax+',ax'+str(i)+str(j)
    # Add a closing bracket to the temporary axis string
    ax+=']'
    # If this axis is the first row, we don't need a comma
    if main_str[-1] == '[':
      main_str+=ax
    # If this axis isn't the first row, we need a comma
    else: main_str = main_str+','+ax
    # Add a final bracket, closing out the main string
  main_str+=']'
  # Initialize axes string and figure string
  fig_str = 'fig,'+main_str+' = plt.subplots('+str(n)+','+str(m)+',figsize=('+str(5*m)+','+str(5*n)+'),linewidth=10)'
  axes_str = 'axes = '+main_str
  # Return the axes and figure strings
  return fig_str,axes_str

def mas_to_rad(mas):
  '''
  Input: A number in miliarcseconds
  Output: A number in radians
  Description: Converts miliarcseconds to radians
  '''
  arcsec = mas / 1000
  degree = arcsec / 3600
  rad = (math.pi * degree) / 180
  return rad

def au_per_pix(pscale,dist):
  '''
  Input: The platescale of a telescope and the distance to an object images with that telescope
  Output: The distance in au per pixel
  Description: Uses the plate scale and distance to an object to calculate the distance per pixel within an image
  '''
  rad = mas_to_rad(pscale)
  pix_dist = (2 * dist) * math.tan(rad/2)
  return pix_dist

def new_out_pathgen(output_path='/content/drive/MyDrive/Follette Lab/Summer 2021/Projects/Transitional Disk Database/PPVII Images/MOCKUPS/Python_Mockups/'):
  '''
  Input: Output path (optional argument)
  Output: A path string that contains a new folder, with an incremented number compared to the previous folder
  Description: Searched within a directory for 'Mockup Round' folders and returns a path string with a newly iterated folder specified
  '''
  folder_list = glob.glob(output_path+'*')

  folder_path = folder_list[-1]
  local_folder = folder_path.split('/')[-1]
  folder_number = local_folder.split(' ')[-1]
  new_folder = 'Mockup Round '+str(int(folder_number)+1)

  new_path = output_path + new_folder

  return(new_path)

def calc_telescopes(mockup):
  '''
  Input: The main mockup dictionary (imported from a csv)
  Output: A dictionary with disk names as keys and telescope names as values
  Description: Returns a dictionary with disk names as keys and telescope names as values
  '''
  # Define empty telescopes dictionary
  telescopes = {}
  # For the length of the names list in mockup
  for i in range(len(mockup['file names'])):
    # Add the name of the telescope to the dictionary, paired with the object's name
    telescopes[mockup['file names'][i]] = mockup['telescope'][i]
  return(telescopes)

def calc_image_paths(folder_names=['VLT','GEMINI','SUBARU']):
  '''
  Input: Optional argument - a list of folder (telescope) names
  Output: A dictionary of image paths keyed to folder/telescope name
  Description: Returns a dictionary of all image paths keyed to their folder/telescope name
  '''
  img_paths = {'Empty':'null'}
  for telescope in folder_names:
    img_paths[telescope] = get_image_paths(telescope)
  return(img_paths)

def process_images(mockup,img_paths):
  '''
  Input: The main mockup dictionary (imported from a csv), a dictionary of telescopes, and a dictionary of image file paths
  Output: The fits data arrays for the images in the mockup
  Description: Returns a dictionary of fits data values keyed by their associated disk name
  '''
  # Define an empty dictionary to be filled and returned
  mockup_images = {}

  # Loop through the mockup names
  for obj in mockup['file names']:
    # For each name, loop through the names of the .fits files
      if obj == 'Empty': mockup_images[obj] = 'Empty'
      for image in img_paths:
      # If there is a match
        if obj in image:
          hdul = fits.open(image)
          data = hdul[0].data
          hdul.close()
          mockup_images[obj] = data
          break
  return(mockup_images)

def scale_bar(mockup):
  '''
  Input: The main mockup dictionary (imported from a csv)
  Output: A dictionary with image name keys and scale bar length values
  Description: Calculates the length in pixels of scale bars on the mockup
  '''
  # Define a dictionary of plate scales for the three telescopes
  plate_scale = {'VLT':12.255,'GEMINI':14.14,'SUBARU':9.5}
  # Define a dictionary keying each mockup image to its telescope
  telescopes = calc_telescopes(mockup)
  # Calculate distance to the object and the distance per pixel using the plate scale
  dist_dict = {}
  dist_per_pixel = {}
  scale_bar_pixels = {}
  for i in range(len(mockup['file names'])):
    try:
      dist_dict[mockup['file names'][i]] = float(((1000 / float(mockup['parallax'][i])) * u.pc).to(u.au) / u.au)
      dist_per_pixel[mockup['file names'][i]] = au_per_pix(plate_scale[telescopes[mockup['file names'][i]]],dist_dict[mockup['file names'][i]])
      scale_bar_pixels[mockup['file names'][i]] = round(50 / dist_per_pixel[mockup['file names'][i]])
    except: ValueError
  return(scale_bar_pixels)

def calc_snip_med(mockup,mockup_images):
  '''
  Input: The main mockup dictionary (imported from a csv) and a dictionary of mockup images
  Output: A dictionary of mockup images with median subtracted backgrounds
  Description: Calculates the median value of image backgrounds to eventually standardize background color
  '''
  # Calculate the median value of a 10% by 10% square on the edge of each image
  snip_med = {}
  for image_name in mockup['file names']:
    dat = mockup_images[image_name]
    dat_snippet = []
    if 'MWC_789' not in image_name and 'HD_100453_GPI_2015-04-10_J' not in image_name:
      for i in range(int(len(dat)/10)):
        dat_snippet_temp = []
        for j in range(int(len(dat)/10)):
          dat_snippet_temp.append(dat[i][j])
        dat_snippet.append(dat_snippet_temp)
        snip_med[image_name] = np.nanmedian(np.concatenate(dat_snippet))
        if np.isnan(snip_med[image_name]): snip_med[image_name] = 0
    else:
      for i in range(int(len(dat))):
        dat_snippet_temp = []
        for j in range(int(len(dat))):
          dat_snippet_temp.append(dat[i][j])
        dat_snippet.append(dat_snippet_temp)
        snip_med[image_name] = np.nanmedian(np.concatenate(dat_snippet))
        if np.isnan(snip_med[image_name]): snip_med[image_name] = 0
  return(snip_med)

def reprocess_images(mockup,mockup_images):
  '''
  Input: The main mockup dictionary (imported from a csv) and a dictionary of mockup images
  Output: A dictionary of mockup images with median subtracted backgrounds
  Description: Subtracts the median value of image backgrounds from the entire image to standardize background color
  '''
  # Find the median value from the mockup images
  snip_med = calc_snip_med(mockup,mockup_images)
  # Subtract that median value from the rest of the image
  for image_name in mockup['file names']:
    if mockup_images[image_name] != 'Empty':
      mockup_images[image_name] = mockup_images[image_name] - snip_med[image_name]
  return(mockup_images)

def make_new_savepath():
  '''
  Input: N/A
  Output: A new savepath
  Description: Calculates, creates, and returns a new file path for the mockup output
  '''
  save_path = new_out_pathgen()
  os.mkdir(save_path)
  return(save_path)

def plot_mockup(mockup,dimensions,colormap,norm_alpha,scale_bar_pixels,save_to_drive=False,save_path='null'):
  '''
  Input: A colormap name, a normalization alpha parameter, a boolean instructing whether to save to drive, and a savepath to export the image
  Output: THE IMAGE
  Description: Prints the image or saves the image to a specified filepath in google drive
  '''
  # Separate width and height values
  nwidth,nheight = dimensions[0],dimensions[1]
  # Create figure and axes
  exec(create_figax_strings(nheight,nwidth)[0],globals())
  exec(create_figax_strings(nheight,nwidth)[1],globals())
  # Set pref
  fig.set_facecolor('black')
  fig.tight_layout(pad=-3.05)
  plt.ioff()

  # Define a dictionary of general image sizes
  im_size = {'VLT':1024,'GEMINI':281,'SUBARU':512}
  # Define two empty dictionaries
  spec_im_dict = {}
  crop_pix = {}
  # Define a dictionary keying each mockup image to its telescope
  telescopes = calc_telescopes(mockup)
  for i in range(len(mockup['names'])):
    # Exception for WRAY15788 - Special dimensions write "FORCE:total_width:crop_radius"
    if 'FORCE' in mockup['crop'][i]:
      spec_im_dict[mockup['names'][i]] = [int(mockup['crop'][i].split(':')[1]),int(mockup['crop'][i].split(':')[2])]
      # spec_im_dict[mockup['names'][i]].append()
      crop_pix[mockup['names'][i]] = int((spec_im_dict[mockup['names'][i]][0] - spec_im_dict[mockup['names'][i]][1]) / 2)
    elif mockup['file names'][i] != 'Empty':
      crop_pix[mockup['names'][i]] = int((im_size[telescopes[mockup['file names'][i]]] - int(mockup['crop'][i])) / 2)
    else: crop_pix['Empty'] = 'Empty'

  for axis in axes:
    for ax in axis: ax.axis('off')

  for i in range(nheight):
    for j in range(nwidth):
      try:
        index = nwidth*i+j

        # Save the survey of the object
        survey = telescopes[mockup['file names'][index]]
        
        img_name = mockup['names'][index]

        # Create dictionaries for each piece of text we want to overplot onto the image
        if 'FORCE' not in mockup['crop'][index]:
          title_dict = {'x':crop_pix[img_name] + ((im_size[survey] - 2*crop_pix[img_name]) * 0.05),'y':crop_pix[img_name] + ((im_size[survey] - 2*crop_pix[img_name]) * 0.93)}
          scale_dict = {'x':crop_pix[img_name] + ((im_size[survey] - 2*crop_pix[img_name]) * 0.05),'y':crop_pix[img_name] + ((im_size[survey] - 2*crop_pix[img_name]) * 0.07)}
          tscope_dict = {'x':crop_pix[img_name] + ((im_size[survey] - 2*crop_pix[img_name]) * 0.05),'y':crop_pix[img_name] + ((im_size[survey] - 2*crop_pix[img_name]) * 0.87)}
          scale_label_dict = {'x':crop_pix[img_name] + ((im_size[survey] - 2*crop_pix[img_name]) * 0.05),'y':crop_pix[img_name] + ((im_size[survey] - 2*crop_pix[img_name]) * 0.1)}
        else:
          title_dict = {'x':crop_pix[img_name] + ((spec_im_dict[img_name][0] - 2*crop_pix[img_name]) * 0.05),'y':crop_pix[img_name] + ((spec_im_dict[img_name][0] - 2*crop_pix[img_name]) * 0.93)}
          scale_dict = {'x':crop_pix[img_name] + ((spec_im_dict[img_name][0] - 2*crop_pix[img_name]) * 0.05),'y':crop_pix[img_name] + ((spec_im_dict[img_name][0] - 2*crop_pix[img_name]) * 0.07)}
          tscope_dict = {'x':crop_pix[img_name] + ((spec_im_dict[img_name][0] - 2*crop_pix[img_name]) * 0.05),'y':crop_pix[img_name] + ((spec_im_dict[img_name][0] - 2*crop_pix[img_name]) * 0.87)}
          scale_label_dict = {'x':crop_pix[img_name] + ((spec_im_dict[img_name][0] - 2*crop_pix[img_name]) * 0.05),'y':crop_pix[img_name] + ((spec_im_dict[img_name][0] - 2*crop_pix[img_name]) * 0.1)}
          print('I LOVE TACOS AND ' + img_name)

        # Create a universal metadata dictionary for overplotted visual elements
        metdat = {'color':'white','fontsize':18,'fontstyle':'oblique','fontweight':'bold'}
        
        # Plot text and visuals over the image, referencing the dictionaries above
        axes[i][j].text(title_dict['x'],title_dict['y'],mockup['names'][index],color=metdat['color'],fontsize=metdat['fontsize'],fontstyle=metdat['fontstyle'],fontweight=metdat['fontweight'])
        axes[i][j].text(tscope_dict['x'],tscope_dict['y'],telescopes[mockup['file names'][index]]+' | '+mockup['band'][index]+'-Band',color=metdat['color'],fontsize=metdat['fontsize']-3,fontstyle=metdat['fontstyle'],fontweight=metdat['fontweight'])
        axes[i][j].text(scale_label_dict['x'],scale_label_dict['y'],'50 AU',color=metdat['color'],fontsize=metdat['fontsize']-6,fontstyle=metdat['fontstyle'],fontweight=metdat['fontweight'])
        axes[i][j].plot([scale_dict['x'],scale_dict['x']+scale_bar_pixels[mockup['file names'][index]]],[scale_dict['y'],scale_dict['y']],linewidth=3,color=metdat['color'])

        # Crop the images based on the universal dictionaries above
        if 'FORCE' not in mockup['crop'][index]:
          axes[i][j].set_xlim(crop_pix[img_name],im_size[survey] - crop_pix[img_name])
          axes[i][j].set_ylim(crop_pix[img_name],im_size[survey] - crop_pix[img_name])
        else:
          axes[i][j].set_xlim(crop_pix[img_name],spec_im_dict[img_name][0] - crop_pix[img_name])
          axes[i][j].set_ylim(crop_pix[img_name],spec_im_dict[img_name][0] - crop_pix[img_name])


        # Pull out the image data
        img = mockup_images[mockup['file names'][index]]
        
        # Check if we want to smooth
        if index in [0,2,3,6,15,16]:
          # We smooth with a Gaussian kernel with x_stddev=1 (and y_stddev=1)
          # It is a 9x9 array
          kernel = Gaussian2DKernel(x_stddev=1)

          # Convolution: scipy's direct convolution mode spreads out NaNs (see
          # panel 2 below)
          img = scipy_convolve(img, kernel, mode='same', method='direct')

        # Log scale the image data (Hyperbolic Arcsine)
        norm = ImageNormalize(mockup_images[mockup['file names'][index]], interval=ManualInterval(0,1), stretch=AsinhStretch(0.1))
        # Define a mesh with colormap and normalization
        axes[i][j].imshow(img,cmap=colormap,norm=norm)
        
        # Plot a circular mask
        if 'FORCE' not in mockup['crop'][index]:
          mask = plt.Circle((im_size[survey]/2,im_size[survey]/2),radius=float(mockup['mask_rad'][index]),color='lightgrey',)
          mask_outline = plt.Circle((im_size[survey]/2,im_size[survey]/2),radius=float(mockup['mask_rad'][index]),color='black',fill=False)
        else:
          mask = plt.Circle((spec_im_dict[img_name][0]/2,spec_im_dict[img_name][0]/2),radius=float(mockup['mask_rad'][index]),color='lightgrey',)
          mask_outline = plt.Circle((spec_im_dict[img_name][0]/2,spec_im_dict[img_name][0]/2),radius=float(mockup['mask_rad'][index]),color='black',fill=False)


        axes[i][j].add_patch(mask)
        axes[i][j].add_patch(mask_outline)

      # Except any KeyErrors (for 'Empty' slots)
      except KeyError:
        axes[i][j].imshow([[0]],cmap = colormap)
  
  # Define Labels and Position Dictionaries
  # xlabel_names = ['M','FGK','AB']
  xlabel_names = ['M','K','G/F','A/B']
  ylabel_names = ['Complex','Featureless','Back Side','Narrow\nShadows','Broad\nShadows','Spirals','Rings']
  xlabel_dict = {'x':np.arange(0,1,1/4) + 1/(2*4),'y':-0.023,'color':'white'}
  ylabel_dict = {'x':-0.023,'y':np.arange(0,1,1/7) + 1/14,'color':'white'}

  for i in range(len(xlabel_names)):
    plt.text(xlabel_dict['x'][i],xlabel_dict['y'], xlabel_names[i], fontsize=35, color=xlabel_dict['color'], transform=plt.gcf().transFigure,horizontalalignment='center')
  for i in range(len(ylabel_names)):
    plt.text(ylabel_dict['x'],ylabel_dict['y'][i], ylabel_names[i], fontsize=35, color=ylabel_dict['color'], transform=plt.gcf().transFigure,horizontalalignment='right')
  
  # Draw borders around the image
  a,b,c,d = -0.005,1.015,-0.005,1.01
  line1 = matplotlib.lines.Line2D((a,b),(c,c),transform=plt.gcf().transFigure,linewidth=4,color='white')
  line2 = matplotlib.lines.Line2D((a,b),(d,d),transform=plt.gcf().transFigure,linewidth=4,color='white')
  line3 = matplotlib.lines.Line2D((a,a),(c,d),transform=plt.gcf().transFigure,linewidth=4,color='white')
  line4 = matplotlib.lines.Line2D((b,b),(c,d),transform=plt.gcf().transFigure,linewidth=4,color='white')

  fig.lines = line1,line2,line3,line4

  # Save the figure to google drive
  if save_to_drive:
    plt.savefig(save_path+'/'+'Mockup '+str(mockup['number'])+' - CMap: '+colormap+' - Alpha: '+str(norm_alpha)+'.png',bbox_inches="tight",facecolor='black')
    plt.close(fig)
