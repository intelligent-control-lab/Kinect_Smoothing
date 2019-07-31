import numpy as np
from functools import partial
from scipy import interpolate, signal
import pykalman

class Crop_Filter(object):
	"""
	The x, y coordinates of the trajectory were captured by some keypoint detection algorithms (e.g. Openpose).
	Sometimes objects will be placed in the background and the depth coordinates may register as invalid values.
	The Crop-Filter crops the invalid values and runs some interpolation methods to replace them.
	"""
	def __init__(self,flag='pchip',min_valid_value=100,max_valid_value=1300):
		"""
		:param flag:  string, specifies the method for crop filtering on the data,
				such as "zero","linear","slinear","quadratic","cubic","previous","next","nearest".
				'pchip': PCHIP 1-d monotonic cubic interpolation, refer to 'Monotone Piecewise Cubic Interpolation'
				'akima': Akima 1D Interpolator, refer to 'A new method of interpolation and smooth curve fitting based on local procedures'
		:param min_valid_value: float, crop-filter the value < min_valid_value
		:param max_valid_value:  float, crop-filter the value > max_valid_value
		"""
		self.flag=flag
		self.min_value = min_valid_value
		self.max_value = max_valid_value

		if flag in ["zero","linear","slinear","quadratic","cubic","previous","next","nearest"]:
			self.filter = partial(interpolate.interp1d,kind=flag)
		elif flag=='pchip' or flag=='PchipInterpolator':
			self.filter = interpolate.PchipInterpolator
		elif flag=='akima' or flag=='Akima1DInterpolator':
			self.filter = interpolate.Akima1DInterpolator

		if flag not in self.all_flags:
			raise('invalid  flags. Only support:', self.all_flags)

	def smooth_trajectory_1d(self,trajectory_1d):
		"""
		smooth the 1-d trajectory or time-series data
		:param trajectory_1d: numpy array, shape of (time_step,)
		:return: numpy-array, smoothed trajectory, same shape as input trajectory_1d
		"""
		valid_ind_s = np.where(trajectory_1d >= self.min_value)[0]
		valid_ind_l = np.where(trajectory_1d <= self.max_value)[0]
		valid_ind = np.intersect1d(valid_ind_s,valid_ind_l,return_indices=False)
		if len(valid_ind)==len(trajectory_1d):
			return trajectory_1d

		t = np.arange(len(trajectory_1d))
		interp_fn = self.filter(valid_ind,trajectory_1d[valid_ind])
		left_ind,right_ind = valid_ind[0],valid_ind[-1]
		smoothed_ind = t[left_ind:right_ind+1]
		smoothed_1d = interp_fn(smoothed_ind) # only interpolate middle are,
		if left_ind>0:
			left_val = trajectory_1d[left_ind]*np.ones(left_ind)
			smoothed_1d = np.concatenate([left_val,smoothed_1d])
		if right_ind<len(trajectory_1d)-1:
			right_val = trajectory_1d[right_ind]*np.ones(len(trajectory_1d)-1-right_ind)
			smoothed_1d = np.concatenate([smoothed_1d,right_val])
		return smoothed_1d

	def smooth_trajectory(self,trajectory):
		"""
		smooth the  trajectory time-series data
		:param trajectory: numpy array, shape of (time_step,coordinate_dim)
		:return: numpy-array, smoothed trajectory, same shape as input trajectory
		"""
		trajectory=np.array(trajectory)
		if len(trajectory.shape)<2:
			trajectory=np.expand_dims(trajectory, axis=1)
		time_step, dim = trajectory.shape[:2]
		smoothed = trajectory.copy()
		for ii in range(dim):
			smoothed[:,ii] = self.smooth_trajectory_1d(trajectory[:,ii])
		return smoothed

	def smooth_multi_trajectories(self,trajectories):
		"""
		smooth the multi-joint trajectories
		:param trajectories: numpy array, shape of (time_step,joint_num, coordinate_dim)
		:return: numpy-array, smoothed trajectories, same shape as input trajectories
		"""
		trajectories=np.array(trajectories)
		if len(trajectories.shape)<3:
			trajectories=np.expand_dims(trajectories, axis=1)
		joint_num = trajectories.shape[1]
		multi_joint_smoothed = trajectories.copy()
		for ii in range(joint_num):
			multi_joint_smoothed[:,ii] = self.smooth_trajectory(trajectories[:,ii])
		return multi_joint_smoothed

	@property
	def all_flags(self):
		flags=[
			"zero",
			"linear",
			"slinear",
			"quadratic",
			"cubic",
			"previous",
			"next",
			"nearest",
			'pchip',#PchipInterpolator
			'PchipInterpolator',
			'akima',#Akima1DInterpolator
		]
		return flags

class GradientCrop_Filter(object):
	"""
	The x, y coordinates of the trajectory were captured by some keypoint detection algorithms (e.g. Openpose).
	Sometimes objects will be placed in the background and the depth coordinates may register as invalid values.
	The GradientCrop_Filter crops the large gradient values maybe miss-classsified as background
	"""
	def __init__(self,flag='pchip',max_valid_gradient=50):
		"""
		:param flag:  string, specifies the method for crop filtering on the data,
				such as "zero","linear","slinear","quadratic","cubic","previous","next","nearest".
				'pchip': PCHIP 1-d monotonic cubic interpolation, refer to 'Monotone Piecewise Cubic Interpolation'
				'akima': Akima 1D Interpolator, refer to 'A new method of interpolation and smooth curve fitting based on local procedures'
		:param max_valid_gradient: float, crop-filter the gradient > max_valid_gradient
		"""
		self.flag=flag
		self.max_valid_gradient = max_valid_gradient

		if flag in ["zero","linear","slinear","quadratic","cubic","previous","next","nearest"]:
			self.filter = partial(interpolate.interp1d,kind=flag)
		elif flag=='pchip' or flag=='PchipInterpolator':
			self.filter = interpolate.PchipInterpolator
		elif flag=='akima' or flag=='Akima1DInterpolator':
			self.filter = interpolate.Akima1DInterpolator

		if flag not in self.all_flags:
			raise('invalid  flags. Only support:', self.all_flags)

	def smooth_trajectory_1d(self,trajectory_1d):
		"""
		smooth the 1-d trajectory or time-series data
		:param trajectory_1d: numpy array, shape of (time_step,)
		:return: numpy-array, smoothed trajectory, same shape as input trajectory_1d
		"""

		valid_ind = []
		valid_value = trajectory_1d[0]
		for ii, val in enumerate(trajectory_1d):
			if abs(valid_value - val) < self.max_valid_gradient :
				valid_value = val
				valid_ind.append(ii)
			elif ii ==len(trajectory_1d)-1:
				valid_ind.append(ii)

		if len(valid_ind)==len(trajectory_1d):
			return trajectory_1d

		t = np.arange(len(trajectory_1d))
		interp_fn = self.filter(valid_ind,trajectory_1d[valid_ind])
		left_ind,right_ind = valid_ind[0],valid_ind[-1]
		smoothed_ind = t[left_ind:right_ind+1]
		smoothed_1d = interp_fn(smoothed_ind) # only interpolate middle are,
		if left_ind>0:
			left_val = trajectory_1d[left_ind]*np.ones(left_ind)
			smoothed_1d = np.concatenate([left_val,smoothed_1d])
		if right_ind<len(trajectory_1d)-1:
			right_val = trajectory_1d[right_ind]*np.ones(len(trajectory_1d)-1-right_ind)
			smoothed_1d = np.concatenate([smoothed_1d,right_val])
		return smoothed_1d

	def smooth_trajectory(self,trajectory):
		"""
		smooth the  trajectory time-series data
		:param trajectory: numpy array, shape of (time_step,coordinate_dim)
		:return: numpy-array, smoothed trajectory, same shape as input trajectory
		"""
		trajectory=np.array(trajectory)
		if len(trajectory.shape)<2:
			trajectory=np.expand_dims(trajectory, axis=1)
		time_step, dim = trajectory.shape[:2]
		smoothed = trajectory.copy()
		for ii in range(dim):
			smoothed[:,ii] = self.smooth_trajectory_1d(trajectory[:,ii])
		return smoothed

	def smooth_multi_trajectories(self,trajectories):
		"""
		smooth the multi-joint trajectories
		:param trajectories: numpy array, shape of (time_step,joint_num, coordinate_dim)
		:return: numpy-array, smoothed trajectories, same shape as input trajectories
		"""
		trajectories=np.array(trajectories)
		if len(trajectories.shape)<3:
			trajectories=np.expand_dims(trajectories, axis=1)
		joint_num = trajectories.shape[1]
		multi_joint_smoothed = trajectories.copy()
		for ii in range(joint_num):
			multi_joint_smoothed[:,ii] = self.smooth_trajectory(trajectories[:,ii])
		return multi_joint_smoothed

	@property
	def all_flags(self):
		flags=[
			"zero",
			"linear",
			"slinear",
			"quadratic",
			"cubic",
			"previous",
			"next",
			"nearest",
			'pchip',#PchipInterpolator
			'PchipInterpolator',
			'akima',#Akima1DInterpolator
		]
		return flags

class Smooth_Filter(object):
	"""
	Smooth the trajectory data
	"""
	def __init__(self,flag='kalman',kernel_size=3, decay_rate=0.6):
		"""
		:param flag: string, specifies the method for smooth filtering,
				'kalman': kalman filter
				'wiener': weiner filter
				'median': median filter
				'moving_average' or 'ma': moving average filter
				'weighted_moving_average' or 'wma': weighted moving average filter
				'exponential_moving_average' or 'ema': exponential moving average
		:param kernel_size: int, kernel size for median filter or wiener filter or moving average filter
		:param decay_rate: float, decay rate for exponential moving average or weighted moving average filter
		"""
		self.flag = flag
		self.kernel_size = kernel_size
		self.decay_rate = decay_rate
		if self.flag=='median':
			self.filter = partial(self._median_filter,kernel_size=kernel_size)
		elif self.flag=='wiener':
			self.filter = partial(self._wiener_filter, kernel_size=kernel_size)
		elif self.flag=='kalman':
			self.filter = self._kalman_filter
		elif self.flag=='moving_average' or self.flag=='ma':
			self.filter = partial(self._ma_filter, kernel_size=kernel_size)
		elif self.flag=='exponential_moving_average' or self.flag=='ema':
			self.filter = partial(self._ema_filter, decay_rate=decay_rate)
		elif self.flag=='weighted_moving_average' or self.flag=='wma':
			self.filter = partial(self._wma_filter, decay_rate=decay_rate)


		if flag not in self.all_flags:
			raise('invalid  flags. Only support:', self.all_flags)

	def _median_filter(self, trajectory,kernel_size):
		"""
		smooth the  time series data with median filter
		:param trajectory: numpy-array
		:param kernel_size: int,  the size of the median filter window
		:return: numpy-array, smoothed time-series data
		"""
		time_step, dim = trajectory.shape[:2]
		filt_traj =trajectory.copy()
		for ii in range(dim):
			filt_traj[:,ii] = signal.medfilt(trajectory[:,ii], kernel_size=kernel_size)
			filt_traj[:, 0] = filt_traj[:, 1]
			filt_traj[:, -1] = filt_traj[:, -2]
		return filt_traj

	def _wiener_filter(self,trajectory,kernel_size):
		"""
		smooth the  time series data with Wiener filter
		:param trajectory: numpy-array
		:param kernel_size: int,  the size of the Wiener filter window
		:return: numpy-array, smoothed time-series data
		"""
		time_step, dim = trajectory.shape[:2]
		filt_traj =trajectory.copy()
		for ii in range(dim):
			filt_traj[:,ii] = signal.wiener(trajectory[:,ii], mysize=kernel_size)
			filt_traj[:, 0] = filt_traj[:, 1]
			filt_traj[:, -1] = filt_traj[:, -2]
		return filt_traj

	def _kalman_filter(self,trajectory):
		"""
		smooth the  time series data with Kalman filter
		:param trajectory: numpy-array
		:return: numpy-array, smoothed time-series data
		"""
		time_step, dim = trajectory.shape[:2]

		self.kf = pykalman.KalmanFilter(n_dim_obs=dim, n_dim_state=dim,initial_state_mean=trajectory[0])
		state_mean, state_covariance = self.kf.filter(trajectory)
		filt_traj = state_mean
		return filt_traj

	def _ma_filter(self,trajectory,kernel_size):
		"""
		smooth the  time series data with moving average
		:param trajectory: numpy-array
		:param kernel_size: int,  the size of the moving average filter window
		:return: numpy-array, smoothed time-series data
		"""
		time_step, dim = trajectory.shape[:2]
		filt_traj =trajectory.copy()
		r = np.arange(1, kernel_size - 1, 2)
		for ii in range(dim):
			a = trajectory[:,ii]
			out0 = np.convolve(a, np.ones(kernel_size, dtype=int), 'valid') / kernel_size
			start = np.cumsum(a[:kernel_size - 1])[::2] / r
			stop = (np.cumsum(a[:-kernel_size:-1])[::2] / r)[::-1]
			filt_traj[:,ii] =  np.concatenate((start, out0, stop))
		return filt_traj

	def _ema_filter(self,trajectory,decay_rate):
		"""
		smooth the  time series data with exponential moving average
		:param trajectory: numpy-array
		:param decay_rate: float,  decay rate for exponential moving average
		:return: numpy-array, smoothed time-series data
		"""
		time_step, dim = trajectory.shape[:2]
		filt_traj =trajectory.copy()
		for ii in range(dim):
			a = trajectory[:,ii]
			smoothed = [a[0]]
			for val in a[1:]:
				new_val = decay_rate * val + (1 - decay_rate) * smoothed[-1]
				smoothed.append(new_val)
			filt_traj[:, ii] = np.array(smoothed)
		return filt_traj

	def _wma_filter(self,trajectory,decay_rate):
		"""
		smooth the  time series data with weighted moving average
		:param trajectory: numpy-array
		:param decay_rate: float,  decay rate for weighted moving average
		:return: numpy-array, smoothed time-series data
		"""
		time_step, dim = trajectory.shape[:2]
		filt_traj =trajectory.copy()
		for ii in range(dim):
			a = trajectory[:,ii]
			smoothed = [a[0]]
			for jj in range(1,len(a)):
				new_val = decay_rate * a[jj] + (1 - decay_rate) * a[jj-1]
				smoothed.append(new_val)
			filt_traj[:, ii] = np.array(smoothed)
		return filt_traj

	def smooth_trajectory(self,trajectory):
		"""
		smooth the  time series data
		:param trajectory: numpy array, shape of (time_step,coordinate_dim)
		:return: numpy-array, smoothed time series data, same shape as input series
		"""
		trajectory=np.array(trajectory)
		if len(trajectory.shape)<2:
			trajectory=trajectory.reshape(trajectory.shape[0],1)
		smoothed=self.filter(trajectory)
		return smoothed

	def smooth_multi_trajectories(self,trajectories):
		"""
		smooth the  multi-joint-trajectories  data
		:param trajectories: numpy array, shape of (time_step,joint_num, coordinate_dim)
		:return: numpy-array, smoothed trajectories, same shape as input trajectories
		"""
		trajectories=np.array(trajectories)
		if len(trajectories.shape)<3:
			trajectories=np.expand_dims(trajectories, axis=1)
		joint_num = trajectories.shape[1]
		multi_joint_smoothed = trajectories.copy()
		for ii in range(joint_num):
			multi_joint_smoothed[:,ii] = self.smooth_trajectory(trajectories[:,ii])
		return multi_joint_smoothed

	@property
	def all_flags(self):
		flags=[
			"median",
			"wiener",
			"kalman",
			"moving_average",
			"ma",
			"exponential_moving_average",
			"ema",
			"weighted_moving_average",
			"wma",
		]
		return flags

class Motion_Sampler(object):
	"""
	For up-sampling or under-sampling in time-series data
	"""
	def __init__(self,motion_threshold=15,min_time_step=30,
	             interpolator='pchip',interpolate_ratio=[3,2,1],interpolate_threshold=[50,100]):
		"""
		:param motion_threshold: float, threshold for motion-detection.
				If x(t) - x(t-1) < motion_threshold, the object considered to be without moving from 't-1' to 't'
		:param min_time_step: int,  minimum remaining time-step after motion-detect under sampling.
		:param interpolator: string, method for interplation for up-sampling the trajectory data,
				such as "zero","linear","slinear","quadratic","cubic","previous","next","nearest".
				'pchip': PCHIP 1-d monotonic cubic interpolation, 'Monotone Piecewise Cubic Interpolation'
				'akima': Akima 1D Interpolator, 'A new method of interpolation and smooth curve fitting based on local procedures'
		:param interpolate_ratio: list of int, interpolation ratio for up-sampling
		:param interpolate_threshold: list of int, interpolation threshold for up-samling
				e.g. if len(data)<interpolate_threshold[0], then ratio = interpolate_ratio[0]
		"""
		self.motion_threshold=motion_threshold
		self.min_time_step = min_time_step
		self.interpolate_ratio = interpolate_ratio
		self.interpolate_threshold = interpolate_threshold

		if interpolator in ["zero","linear","slinear","quadratic","cubic","previous","next","nearest"]:
			self.interpolator = partial(interpolate.interp1d,kind=interpolator)
		elif interpolator=='pchip' or interpolator=='PchipInterpolator':
			self.interpolator = interpolate.PchipInterpolator
		elif interpolator=='akima' or interpolator=='Akima1DInterpolator':
			self.interpolator = interpolate.Akima1DInterpolator

	def motion_detection(self,trajectory,thresh=None):
		"""
		Remove the non-moving part of the trajectory. (Just keep the significant movements)
		:param trajectory: numpy-array, shape of (time_step, *, coordinate_dim)
		:param thresh: float, threshold for motion-detection.
		:return: numpy-array. the new trajectory that filtered out the no-moving-part, have the same shape of the input trajectory
		"""
		if thresh is None:
			thresh = self.motion_threshold
		motion_data = [trajectory[0]]
		left_valid_ind = int(2*len(trajectory)/5)
		right_valid_ind = int(3*len(trajectory)/5)
		for ii in range(1, len(trajectory)):
			diff = np.sum(np.abs(trajectory[ii] - motion_data[-1]))
			if diff >= thresh or (ii>=left_valid_ind and ii<=right_valid_ind):
				motion_data.append(trajectory[ii])
		if len(motion_data) < self.min_time_step and thresh<=0:
			motion_data = self.motion_detection(trajectory, thresh // 2)
		motion_data = np.array(motion_data)
		return motion_data

	def interpolate_trajectory(self,trajectory):
		"""
		interpolate and upsample the trajectory
		:param trajectory: numpy-array, shape of (time_step,  coordinate_dim)
		:return: interpolated trajectory, which has a shape of (interpolate_ratio*time_step,  coordinate_dim)
		"""
		if len(trajectory.shape)<2:
			trajectory=np.expand_dims(trajectory, axis=1)
		L,feat_dim =trajectory.shape[:2]
		t=np.arange(L)

		ratio = self.interpolate_ratio[0]
		for rat, thr in zip (self.interpolate_ratio[1:],self.interpolate_threshold):
			if L>thr:
				ratio=rat
		if ratio<=1:
			return trajectory
		new_t = [ii/ratio for ii in list(range(ratio*(L-1)+1))]
		interp_data=np.zeros((len(new_t),feat_dim))
		for jj in range(feat_dim):
			f = self.interpolator(t, trajectory[:,jj])
			new_traj = f(new_t)
			interp_data[:,jj] = new_traj

		return interp_data

	def interpolate_multi_trajectories(self,trajectories):
		"""
		interpolate and upsample the multi-joint-trajectory
		:param trajectory: numpy-array, shape of (time_step, joint_num,  coordinate_dim)
		:return: interpolated trajectory, which has a shape of (interpolate_ratio*time_step, joint_num, coordinate_dim)
		"""
		if len(trajectories.shape)<3:
			trajectories=np.expand_dims(trajectories, axis=1)

		L,joint_num, feat_dim = trajectories.shape[:3]
		ratio = self.interpolate_ratio[0]
		for rat, thr in zip(self.interpolate_ratio[1:],self.interpolate_threshold):
			if L>thr:
				ratio=rat
		new_t = [ii/ratio for ii in list(range(ratio*(L-1)+1))]
		multi_interpolated=np.zeros((len(new_t),joint_num,feat_dim))
		for ii in range(joint_num):
			multi_interpolated[:, ii] = self.interpolate_trajectory(trajectories[:, ii])
		return multi_interpolated







