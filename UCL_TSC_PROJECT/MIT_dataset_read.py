import scipy.io as scio

mat_data_path = r'C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/MIT Dataset/metadata_blind.mat'
mat_data = scio.loadmat(mat_data_path)
print(mat_data)
metas = dict()

