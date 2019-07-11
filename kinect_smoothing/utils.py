import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def plot_image_frame(image_frame):
	"""
	utils for plot image frames
	:param image_frame: list of images
	"""
	for ii, image in enumerate(image_frame):
		plt.figure()
		if isinstance(image, list):
			image = image[0]
		plt.imshow(image)
		plt.title('frame: ' + str(ii))
		plt.show()


def plot_trajectories(pose_frame):
	"""
	utils for plot trajectory related to time-step t
	:param pose_frame: numpy-array, (time_step,joint_num, ccordinate_dim)
	"""
	pose_frame = np.array(pose_frame)
	timestep, joint_num, dim = pose_frame.shape
	joints = ['neck', 'shoulder', 'elbow', 'hand']
	plt.figure(figsize=(12, 7))
	t = np.arange(timestep)
	for ii, mark in enumerate(joints):
		plt.subplot(331)
		plt.plot(t, pose_frame[:, ii, 0], label=mark)
		plt.xlabel('t')
		plt.ylabel('x')
		plt.subplot(332)
		plt.plot(t, pose_frame[:, ii, 1], label=mark)
		plt.xlabel('t')
		plt.ylabel('y')
		if dim > 2:
			plt.subplot(333)
			plt.plot(t, pose_frame[:, ii, 2], label=mark)
			plt.xlabel('t')
			plt.ylabel('z')
	plt.subplots_adjust(wspace=0.5, hspace=0)
	plt.legend(loc=(1, 0.4))
	plt.show()


def plot_trajectory_3d(trajectory):
	"""
	plot 3d trajectory
	:param trajectory: numpy-array, shape of (time_step,3)
	"""
	xs = trajectory[:, 0]
	ys = trajectory[:, 1]
	zs = trajectory[:, 2]
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.plot3D(xs, ys, zs=zs, marker='o', color='b')
	plt.show()
