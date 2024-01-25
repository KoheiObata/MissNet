import numpy as np
import os

seed=0
np.random.seed(seed)

def make_dir(input_dir):
	if os.path.isdir(input_dir):
		print(f'{input_dir} already exist')
	else:
		os.makedirs(f"{input_dir}")
		print(f'{input_dir} is ready')

def genInvCov(size, low = 0.3 , upper = 0.6, portion = 0.2,symmetric = True):
	portion = portion/2
	A=np.zeros([size,size])
	if size==1:
		return A
	for i in range(size):
		for j in range(size):
			if i>=j:
				continue
			coin=np.random.uniform()
			if coin<portion:
				value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0])
				A[i,j] = value
	if np.allclose(A,np.zeros([size,size])):
		i,j=0,0
		while i==j:
			i=np.random.randint(0,size,1)
			j=np.random.randint(0,size,1)
		value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0])
		A[i,j] = value
	if symmetric:
		A = A + A.T
	return np.matrix(A)

def genInvCov_rectangle(size1, size2, low = 0.3 , upper = 0.6, portion = 0.2):
	portion = portion/2
	A=np.zeros([size1,size2])
	for i in range(size1):
		for j in range(size2):
			# if i>=j:
				# continue
			coin=np.random.uniform()
			if coin<portion:
				value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0])
				A[i,j] = value
			else:
				A[i,j] = 0.01
	if np.allclose(A,np.zeros([size1,size2])):
		i,j=0,0
		while i==j:
			i=np.random.randint(0,size1,1)
			j=np.random.randint(0,size2,1)
		value = (np.random.randint(2) - 0.5)*2*(low + (upper - low)*np.random.rand(1)[0])
		A[i,j] = value
	return np.matrix(A)

def nonlinear_trend(T, beta0=0.2, beta1=60):
	y = np.empty(T)
	for t in range(T):
		y[t] = 1 / (1 + np.exp(beta0*(t - beta1))) + np.random.normal(loc=0, scale=0.3)
	return y

def linear_trend(T):
	alpha = np.random.uniform(0.3,1)
	if np.random.uniform(-1,1) < 0:
		alpha *= -1
	y = np.empty(T)
	for t in range(T):
		y[t] = alpha * t/T
	return y

def sin_seasonal(T, cycles=3):
	# cycles: how many sine cycles
	# T: how many datapoints to generate

	length = np.pi * 2 * cycles
	my_wave = np.sin(np.arange(0, length, length / T))
	return my_wave

def get_latent_d(T, d, cycle_st=1, cycle_ed=20):
	latent_d = np.empty(shape=(T, d))
	for i in range(d):
		y = linear_trend(T)
		y += sin_seasonal(T, cycles=np.random.randint(cycle_st,cycle_ed))[:T]
		y += np.random.normal(loc=0, scale=0.1, size=T)
		latent_d[:, i] = y
	return latent_d

def get_multi_latent_d(latent_d):
	d = latent_d.shape[-1]
	A = genInvCov(d, portion=0.3)
	A += np.eye(d)
	multi_latent_d = np.dot(latent_d, A)
	return multi_latent_d, A

def get_multi_latent_d_rectangle(latent_d, d):
	ld = latent_d.shape[-1]
	A = genInvCov_rectangle(ld, d, portion=0.3)
	multi_latent_d = np.dot(latent_d, A)
	return multi_latent_d, A

def add_multi_latent_d1_d2(multi_latent_d1, multi_latent_d2):
	T, d1 = multi_latent_d1.shape
	T, d2 = multi_latent_d2.shape
	data = np.empty(shape=(T, d1, d2))
	for i in range(d1):
		for j in range(d2):
			data[:, i, j] = np.squeeze(multi_latent_d1[:, i] + multi_latent_d2[:, j])
			data[:, i, j] += np.random.normal(loc=0, scale=0.3, size=T)
	return data



def generate_synthetic(patterns, p_len, d1, latent_d1, savedir=None, num=None):
	n_d1 = np.unique(patterns).shape[0] # number of patterns
	T = np.sum(p_len) # temporal dim

	latent_data = get_latent_d(T, latent_d1)

	multi_latent_d1_list = []
	A_d1_list = []
	for n1 in range(n_d1):
		multi_latent_d1, A_d1 = get_multi_latent_d_rectangle(latent_data, d1)
		multi_latent_d1_list.append(multi_latent_d1)
		A_d1_list.append(A_d1)

	data = np.zeros(shape=(T, d1))
	st, ed = 0, 0
	for p, l in zip(patterns, p_len):
		ed += l
		data[st:ed] = multi_latent_d1_list[p][st:ed]
		st += l

	if savedir:
		path = f'{savedir}/{latent_d1}_{d1}_{T}_{n_d1}'
		if not num==None:
			path += f'/num={num}'
		make_dir(path)
		if os.path.isfile(f'{path}/data.txt'):
			print(path,'data exists')
			return data
		np.savetxt(f'{path}/data.txt',data)
	return data




def experiment(condition):
	PATH='./original/synthetic'
	if condition==0:
		patterns = [0] # patterns
		p_len = [1000] # pattern length
		d1 = 50 # actual dim
		latent_d1 = 10 # latent dim
		savedir = f'{PATH}/pattern'

		num = 5
		for i in range(num):
			data = generate_synthetic(patterns, p_len, d1, latent_d1, savedir, i)

	elif condition==1:
		patterns = [0,1,0,1,0] # patterns
		p_len = [200,200,200,200,200] # pattern length
		d1 = 50 # actual dim
		latent_d1 = 10 # latent dim
		savedir = f'{PATH}/pattern'

		num = 5
		for i in range(num):
			data = generate_synthetic(patterns, p_len, d1, latent_d1, savedir, i)

	elif condition==2:
		patterns = [0] # patterns
		d1 = 50 # actual dim
		latent_d1 = 10 # latent dim
		savedir = f'{PATH}/scale_len'

		for len in [100,250,500,1000,2500,5000,10000,25000,50000,100000,250000,500000]:
			p_len = [len] # pattern length
			data = generate_synthetic(patterns, p_len, d1, latent_d1, savedir)

	elif condition==3:
		patterns = [0] # patterns
		p_len = [1000] # pattern length
		latent_d1 = 10 # latent dim
		savedir = f'{PATH}/scale_dim'

		for d1 in range(20,401,20):
			data = generate_synthetic(patterns, p_len, d1, latent_d1, savedir)

# for condition in range(4):
for condition in range(2):
	experiment(condition)




