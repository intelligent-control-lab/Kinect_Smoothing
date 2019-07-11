import os
from time import time
import joblib
from glob import glob
from multiprocessing import Pool
from functools import partial

from kinect_smoothing import HoleFilling_Filter, Denoising_Filter
from kinect_smoothing import Crop_Filter, Kalman_Filter, Motion_Sampler
from kinect_smoothing import Coordinate_Calculator
from kinect_smoothing import Kinect_Openpose_Pipeline

def simple_pipeline(image_frame,pose_frame):
	pipeline = Kinect_Openpose_Pipeline()
	real_coordinate = pipeline(image_frame,pose_frame)
	return real_coordinate

def standard_pipeline(image_frame,pose_frame):
	hole_filter = HoleFilling_Filter(flag='min')
	image_frame = hole_filter.smooth_image_frames(image_frame)
	noise_filter = Denoising_Filter(flag='modeling', theta=60)
	image_frame = noise_filter.smooth_image_frames(image_frame)

	pose_filter = Crop_Filter(flag='pchip')
	pose_frame = pose_filter.smooth_multi_trajectories(pose_frame)

	calcutator = Coordinate_Calculator()
	raw_pose = calcutator.get_depth_coordinate(image_frame, pose_frame)
	raw_pose = pose_filter.smooth_multi_trajectories(raw_pose)

	smooth_filter = Kalman_Filter()
	raw_pose = smooth_filter.smooth_multi_trajectories(raw_pose)
	real_pose = calcutator.convert_real_coordinate(raw_pose)

	motion_filter = Motion_Sampler(motion_threshold=15, min_time_step=30)
	real_pose = motion_filter.motion_detection(real_pose)

	return real_pose

def kinect_preprocess(img_path, pose_save_dir):
	t1 = time()
	img_path = img_path.replace('\\', '/')
	image_frame = joblib.load(img_path)
	pose_path = img_path.replace('img', 'pose')
	pose_frame = joblib.load(pose_path)

	real_coordinate = standard_pipeline(image_frame,pose_frame)
	#real_coordinate = simple_pipeline(image_frame,pose_frame)

	action, file_name = pose_path.split('/')[:-2]
	new_dir = os.path.join(pose_save_dir, action)
	real_new_path = os.path.join(new_dir, file_name)
	if not os.path.exists(new_dir):
		os.mkdir(new_dir)
	joblib.dump(real_coordinate, real_new_path, compress=3, protocol=2)
	print('preprocessed image %s, time-cost %f s' % (img_path, time() - t1))

def kicet_preprocess_multi(data_dir, pose_save_dir, num_thread=8):
	img_files = glob(data_dir + '*img.pkl')
	print('using multiprocess to preprare smoothing image', num_thread)
	pool = Pool(num_thread)
	pool.map(partial(kinect_preprocess,pose_save_dir=pose_save_dir),img_files)
	print('Processing Done')

if __name__ == '__main__':
	data_dir = 'data/'
	pose_save_dir = 'save_data/'
	if not os.path.exists(pose_save_dir):
		os.mkdir(pose_save_dir)
	kicet_preprocess_multi(data_dir, pose_save_dir, num_thread=8)