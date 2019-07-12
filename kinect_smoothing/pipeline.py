
from .depth_image_smoothing import HoleFilling_Filter, Denoising_Filter
from .trajectory_smoothing import Crop_Filter, Smooth_Filter, Motion_Sampler
from .coordinate_calculation import Coordinate_Calculator

class Kinect_Openpose_Pipeline(object):
	"""
	simple Kinect-Openpose preprocessing pipeline for trajectory prediction
	"""
	def __init__(self,image_holefilling=None, image_denoising=None,
	             tranjectory_crop=None, tranjectory_smooth=None,
	             motion_sampler=None,coordinate_calculator=None):
		"""
		:param image_holefilling: class of HoleFilling_Filter
		:param image_denoising: class of Denoising_Filter
		:param tranjectory_crop: class of Crop_Filter
		:param tranjectory_kalman: class of Smooth_Filter
		:param motion_sampler: class of Motion_Sampler
		:param coordinate_calculator: class of Coordinate_Calculator
		"""
		self.image_holefilling= image_holefilling if image_holefilling is not None else HoleFilling_Filter()
		self.tranjectory_crop = tranjectory_crop if tranjectory_crop is not None else Crop_Filter()
		self.coordinate_calculator = coordinate_calculator if coordinate_calculator is not None else Coordinate_Calculator()
		self.image_denoising = image_denoising
		self.tranjectory_smooth = tranjectory_smooth
		self.motion_sampler = motion_sampler

	def __call__(self, image_frame, openpose_frame):
		"""
		calculate the real world coordinates from kinect-depth images and openpose positions
		:param image_frame: list of numpy array, depth-image frames, [img_0,img_1,...], img_i has a shape of (width, height)
		:param openpose_frame: list of numpy array, position frames, [pose_0,pose_1,...], pose_i has a shape of (joint_num, 2)
		:return: numpy-array, real world coordinates, shape of (time_step, joint_num, 3)
		"""
		image_frame = self.image_holefilling.smooth_image_frames(image_frame)

		if self.image_denoising is not None:
			image_frame = self.image_denoising.smooth_image_frames(image_frame)

		openpose_frame = self.tranjectory_crop.smooth_multi_trajectories(openpose_frame)

		coordinate = self.coordinate_calculator.get_depth_coordinate(image_frame, openpose_frame)

		if self.tranjectory_crop is not None:
			coordinate = self.tranjectory_crop.smooth_multi_trajectories(coordinate)

		if self.tranjectory_smooth is not None:
			coordinate = self.tranjectory_smooth.smooth_multi_trajectories(coordinate)

		coordinate = self.coordinate_calculator.convert_real_coordinate(coordinate)

		if self.motion_sampler is not None:
			coordinate = self.motion_sampler.motion_detection(coordinate)

		return coordinate
