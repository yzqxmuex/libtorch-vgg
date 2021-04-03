// vgg.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <torch/script.h>
#include <torch/torch.h>
#include <torch/nn/pimpl.h>
#include <torch/nn/parallel/data_parallel.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstddef>
#include "args.hxx"
#include "vggnet.hpp"
#include "option.hpp"
#include "cifar10.h"
#include "transform.h"

using namespace torch;
using namespace cv;
using namespace transform;

struct FloatReader
{
	void operator()(const std::string &name, const std::string &value, std::tuple<double, double> &destination)
	{
		size_t commapos = 0;
		std::get<0>(destination) = std::stod(value, &commapos);
		std::get<1>(destination) = std::stod(std::string(value, commapos + 1));
	}
};

COptionMap<int> COptInt;
COptionMap<bool> COptBool;
COptionMap<std::string> COptString;
COptionMap<double> COptDouble;
COptionMap<std::tuple<double, double>> COptTuple;

//为什么要Normalize?
//在这里有两点:第一,VGG网络权重的初始化pytorch和libtorch是一样的,默认使用何凯明初始化,而何凯明初始化很特别的针对具有ReLU激活的网络2/sqrt(n)
//第二,对于数据集的初始化,Normalize((img - mean) / div(std))，如果不进行Normalize则img的值是相当大,在进行反向梯度的时候multiplied by a learning rate
//将会导致W很难收敛,甚至于严重摇晃导致失败
//The goal of applying Feature Scaling is to make sure features are on almost the same scale so that each feature is equally important and make it easier to process by most ML algorithms.
//本例子可以把Normalize去掉就可以发生明显的收敛速度对比
//https://stats.stackexchange.com/questions/185853/why-do-we-need-to-normalize-the-images-before-we-put-them-into-cnn
//链接的文章很好的讲述了为什么要进行Normalize
int main()
{
	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	// Hyper parameters
	const int64_t num_classes = 10;
	const int64_t batch_size = 100;
	const size_t num_epochs = 900;
	const double learning_rate = 0.00001;
	const size_t learning_rate_decay_frequency = 300;  // number of epochs after which to decay the learning rate
	const double learning_rate_decay_factor = 1.0 / 2.0;

	const std::string CIFAR_data_path = "../dataset/cifar10/";
	std::vector<double> norm_mean = { 0.485, 0.456, 0.406 };
	std::vector<double> norm_std = { 0.229, 0.224, 0.225 };
	// CIFAR10 custom dataset
	auto train_dataset = CIFAR10(CIFAR_data_path)
		.map(ConstantPad(4))
		.map(RandomHorizontalFlip())
		.map(RandomCrop({ 32, 32 }))
		.map(torch::data::transforms::Normalize<>(norm_mean, norm_std))
		.map(torch::data::transforms::Stack<>());

	
	auto num_train_samples = train_dataset.size().value();

	auto test_dataset = CIFAR10(CIFAR_data_path, CIFAR10::Mode::kTest)
		.map(torch::data::transforms::Stack<>());

	auto num_test_samples = test_dataset.size().value();

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(train_dataset), batch_size);

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		std::move(test_dataset), batch_size);

	VGG vgg(VGG_CFG::vgg_e, false);
	cout << vgg << endl;
	vgg.to(device);
	
	auto criterion = torch::nn::CrossEntropyLoss();
	criterion->to(device);
	// Optimizer
	torch::optim::Adam optimizer(vgg.parameters(), torch::optim::AdamOptions(learning_rate));

	auto current_learning_rate = learning_rate;

	vgg.train(true);
	for (size_t epoch = 0; epoch != num_epochs; ++epoch)
	{
		double running_loss = 0.0;
		size_t num_correct = 0;

		for (auto& batch : *train_loader)
		{
			// Transfer images and target labels to device
			auto data = batch.data.to(device);
			auto target = batch.target.to(device);
			
			optimizer.zero_grad();
			
			auto output = vgg.forward(data);
			
			auto loss = criterion(output, target);
			//cout << "loss " << loss << endl;

			running_loss += loss.item<double>() * data.size(0);

			auto prediction = output.argmax(1);

			num_correct += prediction.eq(target).sum().item<int64_t>();
			
			loss.backward();
			optimizer.step();
		}
		//printf("k = %d\n", k);
		// Decay learning rate
		if ((epoch + 1) % learning_rate_decay_frequency == 0) {
			current_learning_rate *= learning_rate_decay_factor;
			static_cast<torch::optim::/*SGDOptions*/AdamOptions&>(optimizer.param_groups().front()
				.options()).lr(current_learning_rate);
		}
		auto sample_mean_loss = running_loss / num_train_samples;
		printf("num_correct %d num_train_samples %d\n", num_correct, num_train_samples);
		auto accuracy = static_cast<double>(num_correct) / num_train_samples;

		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
			<< sample_mean_loss << ", Accuracy: " << accuracy << '\n';
	}
}
