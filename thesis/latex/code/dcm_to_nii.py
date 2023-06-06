import os

dcm2niix_path = ".\\tools\\MRIcron\\Resources\\dcm2niix.exe"
dcm2niix_command = "{} -o \"{}\" -b n -z y -f \"{}\" \"{}\""

dcmqi_path = ".\\tools\\dcmqi\\bin\\segimage2itkimage.exe"
dcmqi_command = "{} --inputDICOM \"{}\" --outputDirectory \"{}\" --prefix \"{}\" -t nifti"


ct_target_path = "./dataset/dcm_data/NSCLC1/CT"
ct_output_path = "./dataset/nih_data/NSCLC1/CT"
seg_target_path = "./dataset/dcm_data/NSCLC1/Seg"
seg_output_path = "./dataset/nih_data/NSCLC1/Seg"


def convert_ct():
    for root, dirs, _ in os.walk(ct_target_path, topdown=False):
        for name in dirs:
            os.makedirs(ct_output_path, exist_ok=True)
            os.system(dcm2niix_command.format(
                dcm2niix_path,
                ct_output_path,
                name,
                os.path.join(root, name),
            ))


def convert_seg():
    for root, dirs, files in os.walk(seg_target_path, topdown=False):
        for name in files:
            os.makedirs(seg_output_path, exist_ok=True)
            os.system(dcmqi_command.format(
                dcmqi_path,
                os.path.join(root, name),
                seg_output_path,
                os.path.basename(root)
            ))


def main():
    convert_ct()
    convert_seg()


if __name__ == "__main__":
    main()