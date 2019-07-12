import math
import numpy as np

class Coordinate_Calculator(object):
	"""
	Convert the pixel level coordinate of Kinect to the real world coordinate.
	"""
	def __init__(self,image_width=512,image_height=424,fov_x=70.6,fov_y=60,
	             neighbor_size=3,foreground_max_depth=1200,max_neighbor_size=10,
	             image_pose_delay=0):
		"""
		:param image_width: int, width of depth image
		:param image_height: int, height of depth image
		:param fov_x: float,  horizontal fields of view of depth image
		:param fov_y: float, vertical fields of view of depth image
		:param neighbor_size: int, radius of the neighboring area used for extract depth coordinate
		:param foreground_max_depth: float, max foreground depth for the depth extraction.
				if depth(x,y) > foreground_max_depth, that means (x,y) is in background,
				Then consider the valid depth in neighboring area,
				valid depth =  min(depth[x-neighbor_size:x+neighbor_size,y-neighbor_size:y+neighbor_size])
		:param max_neighbor_size: int, maximum radius of neighboring area
		:param image_pose_delay: int, the delay between image and pose frames
		"""
		self.width = image_width
		self.height = image_height
		self.fov_x = fov_x #fields of view
		self.fov_y = fov_y
		self.f_x = (self.width / 2) / math.tan(0.5 * math.pi * self.fov_x / 180) #focal length of the Kinect camera
		self.f_y = (self.height / 2) / math.tan(0.5 * math.pi * self.fov_y / 180)
		self.neighbor_size = neighbor_size
		self.foreground_max_depth = foreground_max_depth
		self.max_neighbor_size = max_neighbor_size
		self.image_pose_delay = image_pose_delay

	def cal_real_coordinate(self,x,y,z):
		"""
		calculate the real coordinate from pixel level (x,y) and real depth z
		:param x: int or float, pixel level x
		:param y: int or float, pixel level y
		:param z: int or float, real depth
		:return: (real_x, real_y,z),tuple of real word coordinates
		"""
		real_y = (self.height / 2 - y) * z / self.f_y
		real_x = (x - self.width / 2) * z / self.f_x
		return real_x, real_y, z

	def get_depth_coordinate(self,image_frames, position_frames):
		"""
		get depth value from depth-image-frames (from Kinect) and position frames (from Openpose)
		:param image_frames: list of numpy-array, [img_0, img_1,...], img_i has a shape of (height, width)
		:param position_frames: list of numpy-array, [pose_0,pose_1,...], pose_i have a shape of (joint_num, 2)
		:return: numpy array, calculated positions. have a shape of (time_step,joint_num,3)
				for each coordinate (x,y,z), x and y means the pixel level coordinate, and z is real depth
		"""
		delay = self.image_pose_delay
		raw_pose = []
		if delay>0:
			image_frames = image_frames[:-delay]
			position_frames = position_frames[delay:]
		elif delay<0:
			image_frames = image_frames[-delay:]
			position_frames = position_frames[:delay]
		for img, poses in zip(image_frames, position_frames):
			raw_temp = []
			for x, y in poses:
				_radius = self.neighbor_size
				z = 9999999999
				while z >= self.foreground_max_depth and _radius <= self.max_neighbor_size:
					lower_y, lower_x = max(0, y - _radius), max(0, x - _radius)
					z_array = img[lower_y:y + _radius + 1, lower_x:x + _radius + 1]
					z = np.min(z_array)
					_radius += 1
				raw_temp.append(np.array([x, y, z], dtype=np.float32))
			raw_pose.append(raw_temp)
		raw_pose = np.array(raw_pose)
		return raw_pose

	def convert_real_coordinate(self,raw_coordinate):
		"""
		convert raw pixel level (x,y) coordinate and real depth to real world coordinate
		:param raw_coordinate: numpy-array, shape of (time_step, joint_num, 3)
		:return: numpy array, real world coordinate, shape of (time_step, joint_num, 3)
		"""
		real_pose = []
		for poses in raw_coordinate:
			real_temp = []
			for x, y, z in poses:
				real_x,real_y,real_z = self.cal_real_coordinate(x,y,z)
				real_y = round(real_y, 2)
				real_x = round(real_x, 2)
				real_temp.append(np.array([real_x, real_y, real_z], dtype=np.float32))
			real_pose.append(real_temp)
		real_pose = np.array(real_pose)
		return real_pose
