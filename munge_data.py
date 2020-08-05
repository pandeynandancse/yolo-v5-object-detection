import pandas as pd
import os
from sklearn import model_selection
import shutil

DATA_PATH = './wheat/input'
OUTPUT_PATH = './yolo/wheat_data'

def process_data(data,data_type):
	for _ , row in tqdm(data.iterrows(), total = len(data)):
		image_name = row['image_name']
		bounding_boxes = row['bboxes']
		yolo_data = []
		for bbox in bounding_boxes:
			x = bbox[0]
			y = bbox[1]
			w = bbox[2]
			h = bbox[3]
			x_center = x + w / 2 
			y_center = y + h / 2

			#divide by 1024 becoz it is size of image
			x_center /= 1024
			y_center /= 1024
			w /= 1024
			h /= 1024
			#one class in this problem so only 0
			yolo_data.append([0, x_center,y_center,w,h])
		yolo_data = np.array(yolo_data)
		#save np array in txt file
		np.savetxt(
			os.path.join(OUTPUT_PATH, f'labels/{data_type}/{image_name}.txt'),
			yolo_data,
			fmt = ["%d" , "%f", "%f", "%f", "%f"]
			)

		#copy image files from one place to another
		shutil.copyfile(
				#from
				os.path.join(DATA_PATH,f'train/{image_name}.jpg'),
				
				#to
				os.path.join(OUTPUT_PATH,f'images/{data_type}/{image_name}.jpg'),
			)

if __name__ == "__main__":
	df = pd.read_csv(os.path.join(DATA_PATH,'train.csv'))
	df.bbox = df.bbox.apply(ast.literal_eval) #l = '[1,2,3]' ===>>> ast.literal_eval(l) = [1,2,3]
	
	#for every image_id we get list of lists of bounding boxes
	df = df.groupby("image_id")['bbox'].apply(list).reset_index(name='bboxes')

	df_train, df_valid = model_selection.train_test_split(df,tesst_size = 0.1, random_state = 42,shuffle = True)

	df_train = df_train.reset_index(drop=True)
	df_valid = df_valid.reset_index(drop=True)
	#------------------------------------------------------------------------------------------------
	#create folders inside folder wheat_data:
	#images -->> train,validatin
	#labels -->> train,validation
	#labels must be in format like: class, x_center, y_center, width, height inside some_filname.txt
	#labels/trains/many_textfiles_here.txt 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~in short------------------------~~~~~~~~~~~~~~~~
	#here in labels/train folder there is txt file for each training image consists of  all bounding boxes 
	process_data(df_train,data_type = 'train')
	process_data(df_valid,data_type = 'validation')



	#now run python munge_data.py .After this create yaml file