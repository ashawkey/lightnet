#include "cnn.h"

uint32_t byteswap_uint32(uint32_t a)
{
	return ((((a >> 24) & 0xff) << 0) |
		(((a >> 16) & 0xff) << 8) |
		(((a >> 8) & 0xff) << 16) |
		(((a >> 0) & 0xff) << 24));
}

uint8_t* read_file(const char* szFile)
{
	ifstream file(szFile, ios::binary | ios::ate);
	streamsize size = file.tellg();
	file.seekg(0, ios::beg);

	if (size == -1)
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read((char*)buffer, size);
	return buffer;
}

vector<instance*> read_test_cases(int batch_size, int size=60000)
{
	vector<instance*> dataset;

	uint8_t* train_image = read_file("train-images.idx3-ubyte");
	uint8_t* train_labels = read_file("train-labels.idx1-ubyte");

	//uint32_t case_count = byteswap_uint32(*(uint32_t*)(train_image + 4));

	for (int i = 0; i < size / batch_size; i++) {

		tensor<float> X({ batch_size,28,28,1 });
		tensor<float> Y({ batch_size,10 });

		for (int b = 0; b < batch_size; b++) {
			uint8_t* img = train_image + 16 + (i + b) * (28 * 28);
			uint8_t* label = train_labels + 8 + i + b;

			for (int x = 0; x < 28; x++)
				for (int y = 0; y < 28; y++)
					X[{b, x, y, 0}] = img[x + y * 28] / 255.f;

			for (int t = 0; t < 10; t++)
				Y[{b, t}] = *label == t ? 1.0f : 0.0f;
		}
		dataset.push_back(new instance(X, Y));
	}

	delete[] train_image;
	delete[] train_labels;

	return dataset;
}

int main() {

	int MAX_EPOCH = 100;
	float PART = 0.8;
	int DATASET_SIZE = 1000;
	int NUM_CLASSES = 10;
	int BATCH_SIZE = 32;


	cout << "==> Loading data.. " << endl;
	vector<instance*> dataset = read_test_cases(BATCH_SIZE, DATASET_SIZE);
	cout << "    dataset size: " << dataset.size() << endl;
	cout << "==> Loaded! " << endl;

	model cls_machine = model();

	cls_machine.add_layer(new conv2d(1, 3, 3)); // [28, 28, 1] -> [26, 26, 4]
	cls_machine.add_layer(new maxpool2d(2, 2)); // [26, 26, 4] -> [13, 13, 4]
	cls_machine.add_layer(new relu());
	cls_machine.add_layer(new flatten());
	cls_machine.add_layer(new linear(13 * 13 * 3, NUM_CLASSES));

	cls_machine.set_optimizer(new Adam(0.001));
	cls_machine.set_loss(new softmax_cross_entropy_loss(NUM_CLASSES));

	for (int e = 0; e < MAX_EPOCH; e++) {
		cout << "==> EPOCH " << e << endl;
		float total_loss = 0;
		int total_correct = 0;
		int total_case = 0;
		for (int i = 0; i < dataset.size(); i++) {
			instance* in = dataset[i];
			cout << i << endl;
			if (i < dataset.size() * PART) {
				float loss = cls_machine.train_step(in->x, in->y);
				total_loss += loss;
			}
			else {
				tensor<float> logits = softmax(cls_machine.evaluate_step(in->x));
				tensor<int> preds = onehot_to_categorical(logits);
				tensor<int> truths = onehot_to_categorical(in->y);
				//cout << "x: " << in->x << endl;
				//cout << "y: " << in->y << endl;
				//cout << "pred: " << logits << endl;
				for (int b = 0; b < BATCH_SIZE; b++) {
					if (preds[{b, 0}] == truths[{b, 0}]) total_correct++;
					total_case++;
				}
			}
		}
		cout << "    total_loss: " << total_loss << endl;
		cout << "    accuracy: " << total_correct / float(total_case) << " (" << total_correct << "/" << total_case << ")" << endl;
	}
}