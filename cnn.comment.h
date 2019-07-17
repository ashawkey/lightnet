#include <fstream>
#include <cstdint>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <ctime>
#include <random>
#include <cstring>

using namespace std;

template<class T>
class tensor {
/*
    tensor class
    * shape: vector<int>, tensor shape, eg. {3,3} matrix
    * size: int, num of elements, eg. 9 = 3 * 3
*/
public:
	T* data = NULL;
	T* grad = NULL;
	T* grad_old = NULL;
	T* grad_old_square = NULL;

	bool require_grad = false;
	vector<int> shape;
	int size = 0;

	tensor() {}

    // constructor by shape
	tensor(const vector<int>& shape, bool require_grad = false) {
		this->shape = shape;
		this->require_grad = require_grad;
		size = 1;
		for (int i : shape) size *= i;
		data = new T[size];
		memset(data, 0, size * sizeof(T));
		if (require_grad) {
			grad = new T[size];
			grad_old = new T[size];
			grad_old_square = new T[size];
			memset(grad, 0, size * sizeof(T));
			memset(grad_old, 0, size * sizeof(T));
			memset(grad_old_square, 0, size * sizeof(T));
		}
	}

    // copy constructor, call operator=, for parameter passing.
	tensor(const tensor<T>& other) {
		if (this != &other) *this = other;
	}

    // operator=, for assignment.
	tensor<T>& operator=(const tensor<T>& other) {
		shape = other.shape;
		size = other.size;
		require_grad = other.require_grad;
		delete[] data; // delete original data first [!! avoid memory leak !!]
		data = new T[size]; // reallocate
		memcpy(data, other.data, size * sizeof(T));
		if (require_grad) {
			delete[] grad;
			delete[] grad_old;
			delete[] grad_old_square;
			grad = new T[size];
			grad_old = new T[size];
			grad_old_square = new T[size];
			memcpy(grad, other.grad, size * sizeof(T));
			memcpy(grad_old, other.grad_old, size * sizeof(T));
			memcpy(grad_old_square, other.grad_old_square, size * sizeof(T));
		}
		return *this;
	}

    // set gradient from tensor
	void set_grad(const tensor<T>& grad_tensor) {
		assert(require_grad);
		memcpy(grad, grad_tensor.data, size * sizeof(T));
	}

    // random init, inplace, float tensor use only
	void init_randn(float mean = 0, float variance = 1) {
		random_device device;
		default_random_engine generator(device());
		normal_distribution<float> distribution(mean, variance);
		for (int i = 0; i < size; i++)
			data[i] = distribution(generator);
	}

    // xavier normal initialization method
	void init_xavier() {
		int sum_shape = 0;
		for (int i : shape) sum_shape += i;
		float std = sqrt(2.0 / sum_shape);
		init_randn(0, std);
	}

    // index converter
	vector<int> int2vec(int index) const {
		assert(index < size);
		vector<int> res;
		for (int i = shape.size() - 1; i >= 0; i--) {
			if (i == 0) res.push_back(index);
			else {
				res.push_back(index / shape[i]);
				index %= shape[i];
			}
		}
		return res;
	}

    // index converter
	int vec2int(vector<int> index) const {
		assert(index.size() == shape.size());
		int pos = 0;
		int offset = 1;
		for (int i = index.size() - 1; i >= 0; i--) {
			pos += index[i] * offset;
			offset *= shape[i];
		}
		assert(pos < size);
		return pos;
	}

    // index operator, mutable version
	T& operator[] (vector<int> index) {
		assert(index.size() == shape.size());
		for (int i = 0; i < index.size(); i++) index[i] %= shape[i]; // supporting broadcast
		return data[vec2int(index)];
	}

    // index operator, constant version [!! necessary !!]
	T operator[] (vector<int> index) const {
		assert(index.size() == shape.size());
		for (int i = 0; i < index.size(); i++) index[i] %= shape[i]; // supporting broadcast
		return data[vec2int(index)];
	}

    // raw index operator, mutable version
	T& operator[] (int index) {
		// not supporting broadcast, avoid using directly!
		return data[index];
	}

    // raw index operator, constant version
	T operator[] (int index) const {
		return data[index];
	}

    /// arithmetic operator reload
	tensor<T> operator+ (const tensor<T>& other) const {
		assert(shape.size() == other.shape.size());
		if (size == other.size) {
			tensor<T> res(*this);
			for (int i = 0; i < size; i++) res[i] += other[i];
			return res;
		}
		else if (other.size < size) { // very dangerous broadcast
			for (int i = 0; i < shape.size(); i++) assert(shape[i] % other.shape[i] == 0);
			tensor<T> res(*this);
			for (int i = 0; i < size; i++) {
				vector<int> v = int2vec(i);
				res[v] += other[v];
			}
			return res;
		}
		else return other + *this;
	}

	friend tensor<T> operator+ (const tensor<T>& a, const T b) {
		tensor<T> res(a);
		for (int i = 0; i < res.size; i++) res[i] += b;
		return res;
	}

	friend tensor<T> operator+ (const T b, const tensor<T>& a) {
		return a + b;
	}

	tensor<T> operator- (const tensor<T>& other) const {
		assert(shape.size() == other.shape.size());
		if (size == other.size) {
			tensor<T> res(*this);
			for (int i = 0; i < size; i++) res[i] -= other[i];
			return res;
		}
		else if (other.size < size) {
			for (int i = 0; i < shape.size(); i++) assert(shape[i] % other.shape[i] == 0);
			tensor<T> res(*this);
			for (int i = 0; i < size; i++) {
				vector<int> v = int2vec(i);
				res[v] -= other[v];
			}
			return res;
		}
		else return other - *this;
	}

	friend tensor<T> operator- (const tensor<T>& a, const T b) {
		tensor<T> res(a);
		for (int i = 0; i < res.size; i++) res[i] -= b;
		return res;
	}

	friend tensor<T> operator- (const T b, const tensor<T>& a) {
		return a - b;
	}

	tensor<T> operator* (const tensor<T>& other) const {
		assert(shape.size() == other.shape.size());
		if (size == other.size) {
			tensor<T> res(*this);
			for (int i = 0; i < size; i++) res[i] *= other[i];
			return res;
		}
		else if (other.size < size) {
			for (int i = 0; i < shape.size(); i++) assert(shape[i] % other.shape[i] == 0);
			tensor<T> res(*this);
			for (int i = 0; i < size; i++) {
				vector<int> v = int2vec(i);
				res[v] *= other[v];
			}
			return res;
		}
		else return other * *this;
	}

	friend tensor<T> operator* (const tensor<T>& a, const T b) {
		tensor<T> res(a);
		for (int i = 0; i < res.size; i++) res[i] *= b;
		return res;
	}

	friend tensor<T> operator* (const T b, const tensor<T>& a) {
		return a * b;
	}

	tensor<T> operator/ (const tensor<T>& other) const {
		assert(shape.size() == other.shape.size());
		if (size == other.size) {
			tensor<T> res(*this);
			for (int i = 0; i < size; i++) res[i] /= other[i];
			return res;
		}
		else if (other.size < size) {
			for (int i = 0; i < shape.size(); i++) assert(shape[i] % other.shape[i] == 0);
			tensor<T> res(*this);
			for (int i = 0; i < size; i++) {
				vector<int> v = int2vec(i);
				res[v] /= other[v];
			}
			return res;
		}
		else return other / *this;
	}

	friend tensor<T> operator/ (const tensor<T>& a, const T b) {
		tensor<T> res(a);
		for (int i = 0; i < res.size; i++) res[i] /= b;
		return res;
	}

	friend tensor<T> operator/ (const T b, const tensor<T>& a) {
		return a / b;
	}

    // unary minus operator, eg. -t
	tensor<T> operator- () const {
		tensor<T> res(shape);
		for (int i = 0; i < size; i++) res[i] = -data[i];
		return res;
	}

    // convenient output
	friend ostream& operator<<(ostream& out, const tensor<T>& t) {
		out << "<tensor: ";
		for (int i = 0; i < t.shape.size(); i++) {
			if (i == 0) out << "[" << t.shape[i];
			else if (i == t.shape.size() - 1) out << ", " << t.shape[i] << "], ";
			else out << ", " << t.shape[i];
		}
		out << "require_grad = " << t.require_grad << ">" << endl;
		if (t.shape.size() == 2) {
			for (int i = 0; i < t.shape[0]; i++) {
				for (int j = 0; j < t.shape[1]; j++) {
					out << t[{i, j}] << " ";
				}
				out << endl;
			}
		}
		else {
			for (int i = 0; i < t.size; i++) out << t.data[i] << " ";
			out << endl;
		}
		return out;
	}

    /// math operations
	tensor<T> exp() {
		tensor<T> res(*this);
		for (int i = 0; i < size; i++) res[i] = std::exp(res[i]);
		return res;
	}

	tensor<T> log() {
		tensor<T> res(*this);
		for (int i = 0; i < size; i++) res.data[i] = std::log(res.data[i]);
		return res;
	}

    // mimics PyTorch, only change shape.
	tensor<T> view(vector<int> new_shape) {
		int new_size = 1;
		for (int i = 0; i < new_shape.size(); i++) new_size *= new_shape[i];
		assert(size == new_size);
		tensor<float> res(*this);
		res.shape = new_shape;
		return res;
	}

    // transpose matrix
	tensor<T> transpose() {
		assert(shape.size() == 2);
		tensor<T> res({ shape[1], shape[0] });
		for (int i = 0; i < shape[1]; i++) {
			for (int j = 0; j < shape[0]; j++) {
				res[{i, j}] = (*this)[{j, i}];
			}
		}
		return res;
	}

    // matrix multiplication (not supporting batch)
	static tensor<T> matmul(const tensor<T>& a, const tensor<T>& b) {
		assert(a.shape.size() == 2 && b.shape.size() == 2);
		assert(a.shape[1] == b.shape[0]);
		tensor<T> res({ a.shape[0], b.shape[1] });
		for (int i = 0; i < a.shape[0]; i++) {
			for (int j = 0; j < b.shape[1]; j++) {
				for (int k = 0; k < a.shape[1]; k++) {
					res[{i, j}] += a[{i, k}] * b[{k, j}];
				}
			}
		}
		return res;
	}

    // max over axis, retain shape. eg. [3,3] ---max(1)--> [3,1]
	tensor<T> max(int axis) const {
		assert(axis < shape.size());
		vector<int> new_shape(shape);
		new_shape[axis] = 1;
		tensor<T> res(new_shape);
		for (int i = 0; i < size; i++) {
			vector<int> index = int2vec(i);
			index[axis] = 0;
			res[index] = std::max(res[index], data[i]);
		}
		return res;
	}

    // sum over axis
	tensor<T> sum(int axis) const {
		assert(axis < shape.size());
		vector<int> new_shape(shape);
		new_shape[axis] = 1;
		tensor<T> res(new_shape);
		for (int i = 0; i < size; i++) {
			vector<int> index = int2vec(i);
			index[axis] = 0;
			res[index] += data[i];
		}
		return res;
	}

    // max over all elements
	T max() const {
		T res = 0;
		for (int i = 0; i < size; i++) res = max(res, data[i]);
		return res;
	}

	T sum() const {
		T res = 0;
		for (int i = 0; i < size; i++) res += data[i];
		return res;
	}

	T mean() const {
		return sum() / float(size);
	}

    // destructor
	~tensor() {
		delete[] data;
		if (require_grad) {
			delete[] grad;
			delete[] grad_old;
			delete[] grad_old_square;
		}
	}
};


tensor<int> onehot_to_categorical(tensor<float>& x) {
	int batch_size = x.shape[0];
	int num_classes = x.shape[1];
	tensor<int> res({ batch_size, 1 });
	for (int b = 0; b < batch_size; b++) {
		float mx = 0;
		for (int i = 0; i < num_classes; i++) {
			if (x[{b, i}] > mx) {
				mx = x[{b, i}];
				res[{b, 0}] = i;
			}
		}
	}
	return res;
}

class optimizer {
public:
	float lr;
	optimizer() {}
	optimizer(float _lr) {
		lr = _lr;
	}
	virtual void update_weight(tensor<float>& w) = 0;
	/* 
	optimization method reference:
		* http://ruder.io/optimizing-gradient-descent/index.html#stochasticgradientdescent
	*/
};

class SGD : public optimizer {
public:
	float momentum;
	SGD(float _lr = 0.01, float _momentum = 0.9) : optimizer(_lr) {
		momentum = _momentum;
	}
	void update_weight(tensor<float>& w) {
		assert(w.require_grad);
		for (int i = 0; i < w.size; i++) {
			w.grad_old[i] = momentum * w.grad_old[i] + lr * w.grad[i];
			w.data[i] -= w.grad_old[i];
		}
	}
};

class Adam : public optimizer {
public:
	int step;
	float beta1, beta2, epsilon;
	float grad_old_calib, grad_old_square_calib;
	Adam(float _lr = 0.001, float _beta1 = 0.9, float _beta2 = 0.999, float _epsilon = 1e-8) : optimizer(_lr) {
		step = 0;
		beta1 = _beta1;
		beta2 = _beta2;
		epsilon = _epsilon;
	}
	void update_weight(tensor<float>& w) {
		assert(w.require_grad);
		step++;
		for (int i = 0; i < w.size; i++) {
			w.grad_old[i] = beta1 * w.grad_old[i] + (1 - beta1) * w.grad[i];
			w.grad_old_square[i] = beta2 * w.grad_old_square[i] + (1 - beta2) * w.grad[i] * w.grad[i];

			grad_old_calib = w.grad_old[i] / (1 - pow(beta1, step));
			grad_old_square_calib = w.grad_old_square[i] / (1 - pow(beta2, step));

			w.data[i] -= lr * grad_old_calib / (sqrt(grad_old_square_calib) + epsilon);
		}
	}
};


class layer {
public:
	layer() {}
	tensor<float> out;
	tensor<float> grad_out;
	virtual tensor<float> forward(tensor<float>& in) = 0;
	virtual tensor<float> backward(tensor<float>& grad_in, optimizer* optim) = 0;
};

class conv2d : public layer {
public:
	int fin, fout;
	int batch_size, H, W, nH, nW;
	vector<int> kernel_size, stride, padding;
	tensor<float> in;
	tensor<float> weights;
	tensor<float> grad_weights;

	conv2d(int _fin, int _fout, int _kernel_size, int _stride = 1, int _padding = 0) :
		conv2d(_fin, _fout, { _kernel_size, _kernel_size }, { _stride, _stride }, { _padding, _padding }) {}

	conv2d(int _fin, int _fout, vector<int> _kernel_size, vector<int> _stride, vector<int> _padding = vector<int>{ 0,0 }) {
		fin = _fin;
		fout = _fout;
		kernel_size = _kernel_size;
		stride = _stride;
		padding = _padding;

		weights = tensor<float>({ kernel_size[0], kernel_size[1], fin, fout }, true);
		weights.init_xavier();
	}

	tensor<float> forward(tensor<float>& _in) {
		/*
		warning: very naive implementation.
		data shape:
			weights: [K0, K1, fin, fout]
			--> in: [B, H, W, fin]       |    out: [B, nH, nW, fout] -->
			<-- grad_out: [B, H, W, fin] | grad_in: [B, nH, nW, fin] <--
		reference:
			* https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199
		 */
		in = _in;
		batch_size = in.shape[0];
		H = in.shape[1];
		W = in.shape[2];
		nH = (H + 2 * padding[0] - kernel_size[0]) / stride[0] + 1;
		nW = (W + 2 * padding[1] - kernel_size[1]) / stride[1] + 1;
		out = tensor<float>({ batch_size, nH, nW, fout });
		// conv
		for (int b = 0; b < batch_size; b++) {
			for (int i = -padding[0]; i < nH + padding[0]; i += stride[0]) {
				for (int j = -padding[1]; j < nW + padding[1]; j += stride[1]) {
					for (int k = 0; k < fout; k++) {
						// kernel
						for (int ii = 0; ii < kernel_size[0]; ii++) {
							for (int jj = 0; jj < kernel_size[1]; jj++) {
								for (int kk = 0; kk < fin; kk++) {
									int ni = i + ii;
									int nj = j + jj;
									if (ni<0 || nj<0 || ni> H || nj > W) continue;
									else out[{b, i, j, k}] += weights[{ii, jj, kk, k}] * in[{b, ni, nj, kk}];
								}
							}
						}
					}
				}
			}
		}
		return out;
	}

	tensor<float> backward(tensor<float>& grad_in, optimizer* optim) {
		// grad_in: [B, nH, nW, fout]
		grad_out = tensor<float>(in.shape); // [B, H, W, fin] 
		grad_weights = tensor<float>(weights.shape); // [K0, K1, fin, fout]
		// conv
		for (int b = 0; b < batch_size; b++) {
			for (int i = -padding[0]; i < nH + padding[0]; i += stride[0]) {
				for (int j = -padding[1]; j < nW + padding[1]; j += stride[1]) {
					for (int k = 0; k < fout; k++) {
						// kernel
						for (int ii = 0; ii < kernel_size[0]; ii++) {
							for (int jj = 0; jj < kernel_size[1]; jj++) {
								for (int kk = 0; kk < fin; kk++) {
									int ni = i + ii;
									int nj = j + jj;
									if (ni<0 || nj<0 || ni> H || nj > W) continue;
									else {
										grad_out[{b, ni, nj, kk}] += weights[{ii, jj, kk, k}] * grad_in[{b, i, j, k}];
										grad_weights[{ii, jj, kk, k}] += in[{b, ni, nj, kk}] * grad_in[{b, i, j, k}];
									}
								}
							}
						}
					}
				}
			}
		}
		// update weights
		weights.set_grad(grad_weights);
		optim->update_weight(weights);
		return grad_out;
	}
};

class linear : public layer {
public:
	int fin, fout;
	tensor<float> in;
	tensor<float> weights, bias;
	linear(int _fin, int _fout) {
		fin = _fin;
		fout = _fout;

		weights = tensor<float>({ fin, fout }, true);
		bias = tensor<float>({ 1, fout }, true);
		weights.init_xavier();
		bias.init_xavier();
	}

	tensor<float> forward(tensor<float>& _in) {
		in = _in;
		out = tensor<float>::matmul(in, weights) + bias;
		/*
		cout << "linear forward start" << endl;
		cout << "in: " << in << endl;
		cout << "w: " << weights << endl;
		cout << "b: " << bias << endl;
		cout << "out: " << out << endl;
		cout << "linear forward end" << endl;
		*/
		return out;
	}

	tensor<float> backward(tensor<float>& grad_in, optimizer* optim) {
		/*
		data shape:
			--> in: [B, fin]       | out: [B, fout]     -->
			<-- grad_out: [B, fin] | grad_in: [B, fout] <--
		back propagation reference:
			* https://eli.thegreenplace.net/2018/backpropagation-through-a-fully-connected-layer/
		*/
		// grads w.r.t weights & bias
		tensor<float> grad_weights = tensor<float>::matmul(in.transpose(), grad_in);
		tensor<float> grad_bias = grad_in.sum(0);
		weights.set_grad(grad_weights); // [fin, fout]
		bias.set_grad(grad_bias); // [1, fout]
		// update weights
		optim->update_weight(weights);
		optim->update_weight(bias);
		// grads w.r.t inputs
		grad_out = tensor<float>::matmul(grad_in, weights.transpose());
		return grad_out;
	}
};

class flatten : public layer {
public:
	int batch_size;
	tensor<float> in;

	flatten() {}

	tensor<float> forward(tensor<float>& _in) {
		// in: [B, H, W, C]
		in = _in;
		batch_size = in.shape[0];
		out = in.view({ batch_size, in.size / batch_size });
		return out;
	}

	tensor<float> backward(tensor<float>& grad_in, optimizer* optim) {
		grad_out = grad_in.view(in.shape);
		return grad_out;
	}
};

class relu : public layer {
public:
	relu() {}

	tensor<float> forward(tensor<float>& in) {
		out = tensor<float>(in);
		for (int i = 0; i < out.size; i++) {
			if (out[i] < 0) out[i] = 0;
		}
		return out;
	}

	tensor<float> backward(tensor<float>& grad_in, optimizer* optim) {
		grad_out = tensor<float>(grad_in);
		for (int i = 0; i < grad_out.size; i++) {
			if (grad_out[i] < 0) grad_out[i] = 0;
		}
		return grad_out;
	}
};

class maxpool2d : public layer {
public:
	tensor<float> in;
	vector<int> kernel_size;
	vector<int> stride;
	int batch_size;
	int H, W, nH, nW, fin;

	maxpool2d(int _kernel_size, int _stride) : maxpool2d({ _kernel_size, _kernel_size }, { _stride, _stride }) {}

	maxpool2d(vector<int> _kernel_size, vector<int> _stride) {
		kernel_size = _kernel_size;
		stride = _stride;
	}

	tensor<float> forward(tensor<float>& _in) {
		in = _in;
		batch_size = in.shape[0];
		fin = in.shape[3];
		H = in.shape[1];
		W = in.shape[2];
		nH = H / stride[0];
		nW = W / stride[1];
		out = tensor<float>({ batch_size, nH, nW, fin });
		// max pool
		for (int b = 0; b < batch_size; b++) {
			for (int i = 0; i < nH; i += stride[0]) {
				for (int j = 0; j < nW; j += stride[1]) {
					for (int k = 0; k < fin; k++) {
						// kernel
						for (int ii = 0; ii < kernel_size[0]; ii++) {
							for (int jj = 0; jj < kernel_size[1]; jj++) {
								int ni = i + ii;
								int nj = j + jj;
								if (ni<0 || nj<0 || ni > H || nj > W) continue;
								else if (in[{b, ni, nj, k}] > out[{b, i, j, k}]) {
									out[{b, i, j, k}] = in[{b, ni, nj, k}];
								}
							}
						}
					}
				}
			}
		}
		return out;
	}
	tensor<float> backward(tensor<float>& grad_in, optimizer* optim) {
		grad_out = tensor<float>({ batch_size, H, W, fin });
		// max pool
		for (int b = 0; b < batch_size; b++) {
			for (int i = 0; i < nH; i += stride[0]) {
				for (int j = 0; j < nW; j += stride[1]) {
					for (int k = 0; k < fin; k++) {
						// kernel
						for (int ii = 0; ii < kernel_size[0]; ii++) {
							for (int jj = 0; jj < kernel_size[1]; jj++) {
								int ni = i + ii;
								int nj = j + jj;
								if (ni<0 || nj<0 || ni > H || nj > W) continue;
								else if (out[{b, i, j, k}] == in[{b, ni, nj, k}]) {
									grad_out[{b, ni, nj, k}] = grad_in[{b, i, j, k}];
									goto END_WINDOE;
								}
							}
						}
					END_WINDOE:;
					}
				}
			}
		}
		return grad_out;
	}
};

class dropout : public layer {
public:
	float p; // probability to keep
	tensor<bool> hit;

	dropout(float _p = 0.5) {
		p = _p;
	}

	tensor<float> forward(tensor<float>& in) {
		hit = tensor<bool>(in.shape);
		out = tensor<float>(in.shape);
		for (int i = 0; i < in.size; i++) {
			bool active = ((rand() / float(RAND_MAX)) <= p);
			hit[i] = active;
			if (active) out[i] = in[i];
			else out[i] = 0;
		}
		return out;
	}

	tensor<float> backward(tensor<float>& grad_in, optimizer* optim) {
		grad_out = tensor<float>(grad_in.shape);
		for (int i = 0; i < grad_in.size; i++) {
			if (hit[i]) grad_out[i] = grad_in[i];
			else grad_out[i] = 0;
		}
		return grad_out;
	}
};

class loss_function {
public:
	tensor<float> out;
	tensor<float> grad_out;
	virtual tensor<float> forward(tensor<float>& in, tensor<float>& target) = 0;
	virtual tensor<float> backward() = 0;
};

tensor<float> softmax(const tensor<float>& in) {
	// in: [B, f]
	int batch_size = in.shape[0];
	tensor<float> in_exp = (in - in.max(1)).exp();
	tensor<float> res = in_exp / in_exp.sum(1);
	return res;
}

class softmax_cross_entropy_loss : public loss_function {
public:
	int batch_size;
	int num_classes;
	tensor<float> loss;
	tensor<int> target;

	softmax_cross_entropy_loss(int _num_classes) {
		num_classes = _num_classes;
	}

	tensor<float> forward(tensor<float>& in, tensor<float>& _target) {
		// in = [B, f], _target = [B, f] (one_hot)
		this->target = onehot_to_categorical(_target);
		batch_size = in.shape[0];
		// softmax
		out = softmax(in);
		// cross entropy
		loss = tensor<float>({ batch_size, 1 });
		for (int b = 0; b < batch_size; b++)
			loss[{b, 0}] = -log(out[{b, target[{b, 0}]}]);
		return loss;
	}

	tensor<float> backward() {
		/*
		reference:
			* https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative
		*/
		grad_out = tensor<float>({ batch_size, num_classes });
		for (int b = 0; b < batch_size; b++) {
			for (int i = 0; i < num_classes; i++) {
				int y = target[{b, 0}];
				if (i == y) grad_out[{b, i}] = out[{b, i}] - 1;
				else grad_out[{b, i}] = out[{b, i}];
			}
		}
		return grad_out;
	}
};


class l2_loss : public loss_function {
public:
	int batch_size;
	tensor<float> loss;
	tensor<float> target;

	l2_loss() {}

	tensor<float> forward(tensor<float>& in, tensor<float>& target) {
		// in = [B, ...], target = [B, ...]
		this->target = target;
		out = in - target;
		loss = out * out;
		return loss;
	}

	tensor<float> backward() {
		grad_out = 2 * out;
		return grad_out;
	}
};



class model {
public:
	vector<layer*> layers;
	loss_function* loss_layer;
	optimizer* optim;
	tensor<float> loss;
	tensor<float> grad;

	model() {}

	~model() {
		for (layer* l : layers) delete l;
		delete loss_layer;
		delete optim;
	}

	void set_optimizer(optimizer* o) {
		optim = o;
	}
	void set_loss(loss_function* l) {
		loss_layer = l;
	}
	void add_layer(layer* l) {
		layers.push_back(l);
	}

	float train_step(tensor<float>& input, tensor<float>& target) {
		// forward
		tensor<float> x = input;
		for (int i = 0; i < layers.size(); i++) {
			layer* l = layers[i];
			x = l->forward(x);
		}
		// loss function
		loss = loss_layer->forward(x, target);
		grad = loss_layer->backward();
		// backward
		for (int i = layers.size() - 1; i >= 0; i--) {
			layer* l = layers[i];
			grad = l->backward(grad, optim);
		}
		return loss.mean();
	}
	tensor<float> evaluate_step(tensor<float>& input) {
		tensor<float> x = input;
		for (int i = 0; i < layers.size(); i++) {
			layer* l = layers[i];
			x = l->forward(x);
		}
		return x;
	}
};

struct instance {
	tensor<float> x;
	tensor<float> y;
	instance(tensor<float> _x, tensor<float> _y) {
		x = _x;
		y = _y;
	}
};
