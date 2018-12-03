import xlrd
import numpy as np
from glob import glob
import os, os.path
import pydicom
import pickle
########################## READ IN LABELS #######################
patient_names = []
patient_diagnosis = []
LABELS_location = r"G:\Consult Tumors\tcia-diagnosis-data-2012-04-20.xls"

wb = xlrd.open_workbook(LABELS_location)
sheet = wb.sheet_by_index(0)

for i in range(sheet.nrows):
    patient_names.append(sheet.cell_value(i, 0))
    patient_diagnosis.append(sheet.cell_value(i, 1))

# Combine malignant primary lung cancer and malignant metastatic
patient_diagnosis = [2 if x==3 else x for x in patient_diagnosis]
unkown_indecies = np.asarray([i for i, x in enumerate(patient_diagnosis) if x == 0])

# Remove unknowns from patient diagnosis and patient names
for i in unkown_indecies:
    del patient_diagnosis[i]
    del patient_names[i]
    unkown_indecies[:] = [x - 1 for x in unkown_indecies]

patient_names = patient_names[1:]
patient_diagnosis = patient_diagnosis[1:]
patient_diagnosis[:] = [x - 1 for x in list(map(int,patient_diagnosis))]
np.save("G:\\Consult Tumors\\diag",patient_diagnosis)

######################## READ IN IMAGES #########################
image_directory = "G:\\Consult Tumors\\LIDC-IDRI\\*"
# patient_ct_list = glob(image_directory)
# patient_ct_list = [i.split('\\')[-1] for i in patient_ct_list]

patient_ct_list = [image_directory + i for i in patient_names]
total_imagedata = []
largest_stack = 0
for patient in patient_ct_list:
    subdirs = glob(patient + "\\*")
    # 1st Branch
    first_subdir = glob(subdirs[0]+"\\*")[0]
    first_len = len([name for name in os.listdir(first_subdir + "\\")])
    # 2nd Branch
    second_subdir = glob(subdirs[-1] + "\\*")[0]
    second_len = len([name for name in os.listdir(second_subdir + "\\")])

    # If 1st longer then build structure from 1st data
    # If 2nd longer then build structure from 2nd data
    if first_len > second_len:
        patient_image_folder = first_subdir + "\\*dcm"
    else:
        patient_image_folder = second_subdir + "\\*.dcm"
    image_stack = []
    cur_stack_size = 0
    for image in glob(patient_image_folder):
        # Read in each image and construct 3d arrays
        ds = pydicom.dcmread(image)
        image_stack.append(ds.pixel_array)
        cur_stack_size = cur_stack_size + 1
    # Keep track of largest num of slices for zero pad
    if cur_stack_size > largest_stack:
        largest_stack = cur_stack_size
    patient_array = np.asarray(image_stack)
    total_imagedata.append(patient_array)
    print("Patient: " + patient + " added to data array")

with open('outfile', 'wb') as fp:
    pickle.dump(total_imagedata, fp)


largest_stack = 0
for patient in total_imagedata:
    if(len(np.shape(patient)) < 4):
        if(np.shape(patient)[0] > largest_stack):
            largest_stack = np.shape(patient)[0]
    else:
        p = patient[0]
        if (np.shape(p)[0] > largest_stack):
            largest_stack = np.shape(p)[0]
print(largest_stack)
print("Generating zero padded list")
largest_stack = 545
newtotaldata = []
c = 0
for patient in total_imagedata:
    c = c + 1
    if (len(np.shape(patient)) < 4):
        if (np.shape(patient)[0] < largest_stack):
            print(c)
            tmp = np.pad(patient, ((0, largest_stack - np.shape(patient)[0]), (0, 0), (0, 0)), 'constant').astype(np.uint8)
            tmp[tmp < 0 ] = 0
        else:
            tmp = patient
            tmp[tmp < 0] = 0
        newtotaldata.append(tmp)
    else:
        p = patient[0]
        if (np.shape(p)[0] < largest_stack):
            print(c)
            tmp = np.pad(p, ((0, largest_stack - np.shape(p)[0]), (0, 0), (0, 0)), 'constant').astype(np.uint8)
            tmp[tmp < 0] = 0
        else:
            tmp = p
            tmp[tmp < 0] = 0
        newtotaldata.append(tmp)
# print("saving list")
# with open('G:\\Consult Tumors\\paddedlist', 'wb') as fp:
#    pickle.dump(newtotaldata, fp)
del total_imagedata[:]
newtotaldata = np.array(newtotaldata)
print(np.shape(newtotaldata))
fp = np.memmap("G:\\Consult Tumors\\paddedArray.dat", mode='w+', dtype='float16', shape=np.shape(newtotaldata))
fp[:] = newtotaldata[:]
print("saving array")