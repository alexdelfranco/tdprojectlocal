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

def get_sheet_data(wb):
  sheet = wb.worksheet('For Python')
  data = sheet.get_all_values()
  df = pd.DataFrame(data)
  # Arrange Pandas dataframe
  df.columns = df.iloc[0]
  df = df.drop(df.index[0])
  df = df.reset_index()
  return df

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
  col_names['labels'] = 'Mockup '+str(mockup_number)+' Labels'
  col_names['smoothing'] = 'Mockup '+str(mockup_number)+' Smoothing'

  # Initialize a dicitonary to return
  columns = {}
  # Determine the length of each column
  for i in range(len(dataframe[col_names['names']])):
    if dataframe[col_names['names']][i] == '':
      columns['length'] = i
      break
  # Determine the number of rows
  labels = []
  for i in range(len(dataframe[col_names['labels']])):
    if dataframe[col_names['labels']][i] == '':
      columns['label count'] = i
      break
    else:
      labels.append(dataframe[col_names['labels']][i])
  columns['labels'] = labels
  
  # For each of the eight columns of data
  for column in ['names','file names','telescope','parallax','mask_rad','crop','band','smoothing']:
    columns[column] = dataframe[col_names[column]].values.tolist()[0:columns['length']]
  # Add mockup number
  columns['number'] = mockup_number

  return columns

def mlist_extract(dataframe, total_mockup_number):
  mockups = {}  

  for mockup_number in range(1,total_mockup_number+1):
    mockup = mockup_extract(dataframe,mockup_number)
    # Add to return dictionary
    mockups['Mockup '+str(mockup_number)] = mockup
  
  mockups['number'] = total_mockup_number
  return mockups

def set_mockup(df,total_mockups,mockup_number):
  mockups = mlist_extract(df, total_mockups)
  mockup = mockups['Mockup '+str(mockup_number)]
  return mockup

def orig_pathlist(mockup):
  file_names = mockup['file names']
  # Initialize empty list for the image data
  path_list = []
  # Loop through the telescope folders
  for folder in ['VLT','GEMINI','SUBARU']:
    # Find the image paths for each folder and append them to the initialized list
    image_paths = get_image_paths('Original_Data/'+folder)
    # Check to see if the paths have already been appended
    for path in image_paths:
      for name in file_names:
        # If they haven't been, then append
        if name in path:
          path_list.append(path)
  return path_list

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

def mask_imsize_crop_calc(mockup):

  im_size = {'VLT':1024,'GEMINI':281,'SUBARU':512}
  spec_im_dict = {}
  crop_pix = {}
  image_size = {}
  mask_rad = {}

  for i in range(len(mockup['names'])):
    # Exception for special dimensions write "FORCE:total_width:crop_radius"
    if 'FORCE' in mockup['crop'][i]:
      spec_im_dict[mockup['names'][i]] = [int(mockup['crop'][i].split(':')[1]),int(mockup['crop'][i].split(':')[2])]
      crop_pix[mockup['names'][i]] = int((spec_im_dict[mockup['names'][i]][0] - spec_im_dict[mockup['names'][i]][1]) / 2)
    elif mockup['file names'][i] != 'Empty':
      crop_pix[mockup['names'][i]] = int((im_size[mockup['telescope'][i]] - int(mockup['crop'][i])) / 2)
    else: crop_pix['Empty'] = 'Empty'

  for index in range(0,mockup['length']):
    img_name = mockup['names'][index]
    if 'FORCE' not in mockup['crop'][index]:
      image_size[mockup['names'][index]] = im_size[mockup['telescope'][index]]
    else:
      image_size[mockup['names'][index]] = spec_im_dict[img_name][0]
    mask_rad[mockup['names'][index]] = float(mockup['mask_rad'][index])

  return {'Mask Radii':mask_rad,'Image Sizes':image_size,'Edge Crop':crop_pix}

def mask_remove(matrix,mockup,image_name,mask_imsize_crop):
  image_size,mask_radius = mask_imsize_crop['Image Sizes'],mask_imsize_crop['Mask Radii']
  # Defines the radius of pixels to remove from the image center (reduced by a percentage so there is overlap)
  r = mask_radius[image_name] * 0.85
  for i in range(-1*int(len(matrix)/2),int(len(matrix)/2)):
    for j in range(-1*int(len(matrix)/2),int(len(matrix)/2)):
      if (i**2 + j**2 < r**2):
        # Set pixel value to nan
        matrix[i + int(len(matrix)/2)][j + int(len(matrix)/2)] = np.nan
  return(matrix)

def edge_remove(matrix,mockup,image_name,mask_imsize_crop):
  # Defines the radius of pixels to remove from the image edge (reduced by a percentage so there is overlap)
  edge_crop = int(mask_imsize_crop['Edge Crop'][image_name] * 0.95)
  matrix[-1*edge_crop],matrix[:edge_crop],matrix[:,-1*edge_crop],matrix[:,:edge_crop] = np.nan,np.nan,np.nan,np.nan
  return(matrix)

def hdul_norm(image,slice_dict,mockup,image_name,mask_imsize_crop):
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
  
  # A CAREFUL CHANGE
  image_cut = edge_remove(image_cut,mockup,image_name,mask_imsize_crop)
  image_cut = mask_remove(image_cut,mockup,image_name,mask_imsize_crop)
 
  # Set 0.03 percentile of the image to 0
  image_cut -= np.nanpercentile(image_cut,0.03)
  # Normalize the image to the 99.975th percentile
  norm_dat = image_cut/np.nanpercentile(image_cut,99.975)

  snp_med = snip_med(image_name,norm_dat,mockup,mask_imsize_crop)
  image_cut_sub = norm_dat - snp_med

  # Append the new image to the newly created fits header object
  hdul_new.append(fits.ImageHDU(data=image_cut_sub))
  
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

def add_backslash(path):
  path_arr = path.split(' ')
  full_path = ''
  for segment in path_arr:
    full_path += segment
    full_path += '\ '
  return full_path[:-2]

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

def im_name(file_name,mockup):
  for i in range(len(mockup['file names'])):
    if mockup['file names'][i] == file_name:
      return mockup['names'][i]

def process_fits(mockup,slice_dict,force_write=False):
  '''
  Input: A list of .fits file paths
  Output: Returns nothing, writes an output of fully processed .fits files to a specified filepath
  Description: Writes the normalized and scaled .fits files to a folder in google drive
  '''
  orig_paths = orig_pathlist(mockup)
  processed_image_paths = get_image_paths('Processed Images')
  mockup_paths = {}
  mockup_images = {}
  for file_name in tqdm(mockup['file names']):
    duplicate = False
    
    for path in orig_paths:
      if file_name in path:
        mockup_paths[file_name] = path

    for path in processed_image_paths:
      if file_name in path:
        duplicate = True
        if force_write:
          # Remove that file and replace it with a new one
          path = add_backslash(path)
          !rm {path}
        else:
          # Save to mockup image dict
          hdul = fits.open(path)
          data = hdul[0].data
          hdul.close()
          mockup_images[file_name] = data
        break
    
    if (not duplicate) or (duplicate and force_write):
      # Initialize a normalized, extreme-cut fits header object
      hdul = hdul_norm(mockup_paths[file_name],slice_dict,mockup,im_name(file_name,mockup),mask_imsize_crop_calc(mockup))
      # Create a new string path for the new .fits file
      write_path = insert_path(mockup_paths[file_name],'Processed Images',-1)
      # Write the new .fits file data into a file
      hdul.writeto(write_path)
      # Save to mockup image dict
      data = hdul[0].data
      hdul.close()
      mockup_images[file_name] = data
  return mockup_images

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

def scale_bar(mockup):
  '''
  Input: The main mockup dictionary (imported from a csv)
  Output: A dictionary with image name keys and scale bar length values
  Description: Calculates the length in pixels of scale bars on the mockup
  '''
  # Define a dictionary of plate scales for the three telescopes
  plate_scale = {'VLT':12.255,'GEMINI':14.14,'SUBARU':9.5}
  # Calculate distance to the object and the distance per pixel using the plate scale
  dist_dict = {}
  dist_per_pixel = {}
  scale_bar_pixels = {}
  for i in range(len(mockup['file names'])):
    try:
      dist_dict[mockup['file names'][i]] = float(((1000 / float(mockup['parallax'][i])) * u.pc).to(u.au) / u.au)
      dist_per_pixel[mockup['file names'][i]] = au_per_pix(plate_scale[mockup['telescope'][i]],dist_dict[mockup['file names'][i]])
      scale_bar_pixels[mockup['file names'][i]] = round(50 / dist_per_pixel[mockup['file names'][i]])
    except: ValueError
  return(scale_bar_pixels)

def snip_med(image_name,image_data,mockup,mask_imsize_crop):
  '''
  Needs to be updated
  Input: The main mockup dictionary (imported from a csv) and a dictionary of mockup images
  Output: A dictionary of mockup images with median subtracted backgrounds
  Description: Calculates the median value of image backgrounds to eventually standardize background color
  '''
  dat_snippet = []
  image_size,edge_crop = mask_imsize_crop['Image Sizes'],mask_imsize_crop['Edge Crop']

  if 'MWC_789' not in image_name and 'HD_100453_GPI_2015-04-10_J' not in image_name:
    for i in range(int(edge_crop[image_name]),int(edge_crop[image_name]+((image_size[image_name]-edge_crop[image_name])/2)/10)):
      dat_snippet_temp = []
      for j in range(int(edge_crop[image_name]),int(edge_crop[image_name]+((image_size[image_name]-edge_crop[image_name])/2)/10)):
        dat_snippet_temp.append(image_data[i][j])
      dat_snippet.append(dat_snippet_temp)
    snip_med = np.nanmedian(np.concatenate(dat_snippet))
    if np.isnan(snip_med): snip_med = 0
  else:
    for i in range(int(edge_crop[image_name]),int(edge_crop[image_name]+((image_size[image_name]-edge_crop[image_name])/2)/10)):
      dat_snippet_temp = []
      for j in range(int(edge_crop[image_name]),int(edge_crop[image_name]+((image_size[image_name]-edge_crop[image_name])/2)/10)):
        dat_snippet_temp.append(image_data[i][j])
      dat_snippet.append(dat_snippet_temp)
    snip_med = np.nanmedian(np.concatenate(dat_snippet))
    if np.isnan(snip_med): snip_med = 0
  return(snip_med)

def make_new_savepath():
  '''
  Input: N/A
  Output: A new savepath
  Description: Calculates, creates, and returns a new file path for the mockup output
  '''
  save_path = new_out_pathgen()
  os.mkdir(save_path)
  return(save_path)

def save_image(save_to_drive=False):
  # Create a new save path
  if save_to_drive:
    save_path = make_new_savepath()
    print('Save Path: '+save_path)
  else: save_path = 'null'
  return save_path

def plot_mockup(colormap,norm_alpha,mockup,mockup_images,save_path):
  '''
  Input: A colormap name, a normalization alpha parameter, a boolean instructing whether to save to drive, and a savepath to export the image
  Output: THE IMAGE
  Description: Prints the image or saves the image to a specified filepath in google drive
  '''
  # Separate width and height values
  nwidth,nheight = int(mockup['length']/mockup['label count']),int(mockup['label count'])
  # Create figure and axes
  exec(create_figax_strings(nheight,nwidth)[0],globals())
  exec(create_figax_strings(nheight,nwidth)[1],globals())
  # Set specific figure preferences
  fig.set_facecolor('black')
  fig.tight_layout(pad=-3.05)
  plt.ioff()

  # Define some helpful data dictionaries
  mimc = mask_imsize_crop_calc(mockup)
  im_size,crop_pix,mask_pix = mimc['Image Sizes'],mimc['Edge Crop'],mimc['Mask Radii']

  # Define a dictionary keying each mockup image to a scale bar length
  scale_bar_pixels = scale_bar(mockup)

  for axis in axes:
    for ax in axis: ax.axis('off')

  for i in range(nheight):
    for j in range(nwidth):
      try:
        index = nwidth*i+j
        img_name = mockup['names'][index]

        # Create dictionaries for each piece of text we want to overplot onto the image
        title_dict = {'x':crop_pix[img_name] + ((im_size[img_name] - 2*crop_pix[img_name]) * 0.05),'y':crop_pix[img_name] + ((im_size[img_name] - 2*crop_pix[img_name]) * 0.93)}
        scale_dict = {'x':crop_pix[img_name] + ((im_size[img_name] - 2*crop_pix[img_name]) * 0.05),'y':crop_pix[img_name] + ((im_size[img_name] - 2*crop_pix[img_name]) * 0.07)}
        tscope_dict = {'x':crop_pix[img_name] + ((im_size[img_name] - 2*crop_pix[img_name]) * 0.05),'y':crop_pix[img_name] + ((im_size[img_name] - 2*crop_pix[img_name]) * 0.87)}
        scale_label_dict = {'x':crop_pix[img_name] + ((im_size[img_name] - 2*crop_pix[img_name]) * 0.05),'y':crop_pix[img_name] + ((im_size[img_name] - 2*crop_pix[img_name]) * 0.1)}
        
        # Create a universal metadata dictionary for overplotted visual elements
        metdat = {'color':'white','fontsize':18,'fontstyle':'oblique','fontweight':'bold'}
        
        # Plot text and visuals over the image, referencing the dictionaries above
        axes[i][j].text(title_dict['x'],title_dict['y'],mockup['names'][index],color=metdat['color'],fontsize=metdat['fontsize'],fontstyle=metdat['fontstyle'],fontweight=metdat['fontweight'])
        axes[i][j].text(tscope_dict['x'],tscope_dict['y'],mockup['telescope'][index]+' | '+mockup['band'][index]+'-Band',color=metdat['color'],fontsize=metdat['fontsize']-3,fontstyle=metdat['fontstyle'],fontweight=metdat['fontweight'])
        axes[i][j].text(scale_label_dict['x'],scale_label_dict['y'],'50 AU',color=metdat['color'],fontsize=metdat['fontsize']-6,fontstyle=metdat['fontstyle'],fontweight=metdat['fontweight'])
        axes[i][j].plot([scale_dict['x'],scale_dict['x']+scale_bar_pixels[mockup['file names'][index]]],[scale_dict['y'],scale_dict['y']],linewidth=3,color=metdat['color'])

        # Crop the images based on the universal dictionaries above
        axes[i][j].set_xlim(crop_pix[img_name],im_size[img_name] - crop_pix[img_name])
        axes[i][j].set_ylim(crop_pix[img_name],im_size[img_name] - crop_pix[img_name])
        
        # Pull out the image data
        img = mockup_images[mockup['file names'][index]]
        
        # Check if we want to smooth
        if mockup['smoothing'][index] == 'TRUE':
          # We smooth with a Gaussian kernel with x_stddev=1 (and y_stddev=1)
          # It is a 9x9 array
          kernel = Gaussian2DKernel(x_stddev=1)
          print('WE SMOOTHED IMAGE NUMBER '+str(index))

          # Convolution: scipy's direct convolution mode spreads out NaNs (see
          # panel 2 below)
          img = scipy_convolve(img, kernel, mode='same', method='direct')

        # Log scale the image data (Hyperbolic Arcsine)
        norm = ImageNormalize(mockup_images[mockup['file names'][index]], interval=ManualInterval(0,1), stretch=AsinhStretch(norm_alpha))
        # Define a mesh with colormap and normalization
        axes[i][j].imshow(img,cmap=colormap,norm=norm)
        
        # Plot a circular mask
        mask = plt.Circle((im_size[img_name]/2,im_size[img_name]/2),radius=float(mask_pix[img_name]),color='lightgrey',)
        mask_outline = plt.Circle((im_size[img_name]/2,im_size[img_name]/2),radius=float(mask_pix[img_name]),color='black',fill=False)

        axes[i][j].add_patch(mask)
        axes[i][j].add_patch(mask_outline)

      # Except any KeyErrors (for 'Empty' slots)
      except KeyError:
        axes[i][j].imshow([[0]],cmap = colormap)
  
  # Define Labels and Position Dictionaries
  xlabel_names = ['M','K','G/F','A/B']
  ylabel_names = mockup['labels'][::-1]
  xlabel_dict = {'x':np.arange(0,1,1/4) + 1/(2*4),'y':-0.023,'color':'white'}
  ylabel_dict = {'x':-0.023,'y':np.arange(0,1,1/mockup['label count']) + 1/(2*mockup['label count']),'color':'white'}

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
  
  if save_path != 'null':
    plt.savefig(save_path+'/'+'Mockup '+str(mockup['number'])+' - CMap: '+colormap+' - Alpha: '+str(norm_alpha)+'.png',bbox_inches="tight",facecolor='black')
    plt.close(fig)
