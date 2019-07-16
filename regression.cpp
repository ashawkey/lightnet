#include "cnn.h"

int main(){
	
	// simple regression

    int MAX_EPOCH = 100;
	int DATASET_LENGTH = 100;
	float PART = 0.5;

	vector<instance*> dataset;
	cout << "==> Generating regression data... " << endl;
	for (int i = 0; i < DATASET_LENGTH; i++) {
		auto x = tensor<float>({ 1, 4 });
		auto y = tensor<float>({ 1, 1 });
		x.init_randn();
		y[0] = (x*x).sum();
		dataset.push_back(new instance(x, y));
	}
	cout << "==> Generated! " << endl;

	model reg_machine = model();
	reg_machine.add_layer(new linear(4, 16));
	reg_machine.add_layer(new relu());
	reg_machine.add_layer(new linear(16, 1));
	reg_machine.set_optimizer(new Adam(0.001));
	reg_machine.set_loss(new l2_loss());

	for (int e = 0; e < MAX_EPOCH; e++) {
		cout << "==> EPOCH " << e << endl;
		float total_loss = 0;
		float l2_error = 0;
		for (int i = 0; i < dataset.size(); i++) {
			auto in = dataset[i];
			if (i < dataset.size() * PART) {
				float loss = reg_machine.train_step(in->x, in->y);
				total_loss += loss;
			}
			else {
				tensor<float> logits = reg_machine.evaluate_step(in->x);
				tensor<float> error = (logits - in->y);
				//cout << "y: " << in->y << endl;
				//cout << "pred: " << logits << endl;
				l2_error += (error * error).sum();
			}
		}
		cout << "    total_loss: " << total_loss << endl;
		cout << "    l2_error: " << l2_error << endl;
	}
}