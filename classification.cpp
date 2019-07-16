#include "cnn.h"

using namespace std;

int main() {

	// simple classification

	int MAX_EPOCH = 100;
	int DATASET_LENGTH = 200;
	float PART = 0.5;
	int NUM_CLASSES = 3;
	int IN_CHANNEL = 2;

	vector<instance*> dataset;
	cout << "==> Generating classification data... " << endl;
	for (int i = 0; i < DATASET_LENGTH; i++) {
		auto x = tensor<float>({ 1, IN_CHANNEL });
		auto noise = tensor<float>(x.shape);
		auto y = tensor<float>({ 1, NUM_CLASSES });
		x.init_randn(0, 2);
		noise.init_randn(0, 0.01);
		int tmp = sqrt((x*x).sum());
		if (tmp < 1) y[0] = 1;
		else if (tmp > 3) y[2] = 1;
		else y[1] = 1;
		dataset.push_back(new instance(x + noise, y));
	}
	cout << "==> Generated! " << endl;

	model cls_machine = model();
	cls_machine.add_layer(new linear(IN_CHANNEL, 16));
	cls_machine.add_layer(new relu());
	cls_machine.add_layer(new linear(16, NUM_CLASSES));
	cls_machine.set_optimizer(new Adam(0.001));
	cls_machine.set_loss(new softmax_cross_entropy_loss(NUM_CLASSES));

	for (int e = 0; e < MAX_EPOCH; e++) {
		cout << "==> EPOCH " << e << endl;
		float total_loss = 0;
		float total_correct = 0;
		float total_case = 0;
		for (int i = 0; i < dataset.size(); i++) {
			instance* in = dataset[i];
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
				if (preds[0] == truths[0]) total_correct++;
				total_case++;
			}
		}
		cout << "    total_loss: " << total_loss << endl;
		cout << "    accuracy: " << total_correct / float(total_case) << " (" << total_correct << "/" << total_case << ")" << endl;
	}
}