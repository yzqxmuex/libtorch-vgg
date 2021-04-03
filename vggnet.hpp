#include<torch/script.h>
#include <torch/torch.h>
#include <map> 

enum VGG_CFG
{
	vgg_a = 0,
	vgg_b,
	vgg_d,
	vgg_e,
} ;

int vggCfg[4][21] = { {64, 2, 128, 2, 256, 256, 2, 512, 512, 2, 512, 512, 2},
					{64, 64, 2, 128, 128, 2, 256, 256, 2, 512, 512, 2, 512, 512, 2},
					{64, 64, 2, 128, 128, 2, 256, 256, 256, 2, 512, 512, 512, 2, 512, 512, 512, 2},
					{64, 64, 2, 128, 128, 2, 256, 256, 256, 256, 2, 512, 512, 512, 512, 2, 512, 512, 512, 512, 2} };

struct VGG : public torch::nn::Module
{
	VGG(VGG_CFG cfg, bool nBN): in_channels(3)
	{
		make_model(cfg, nBN);
		featuresModel = register_module("features", features.ptr());

		//VGG 分类器,对于CIFAR10,最后10个分类
		torch::nn::Sequential classifier(
			torch::nn::Dropout(),
			torch::nn::Linear(512, 512),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Dropout(),
			torch::nn::Linear(512, 512),
			torch::nn::ReLU(torch::nn::ReLUOptions(true)),
			torch::nn::Linear(512, 10));
		classifierModel = register_module("classifier", classifier.ptr());
	}
	
	~VGG()
	{

	}

	torch::Tensor forward(torch::Tensor input)
	{
		//y = featuresModel.ptr()->children().at(5)->forward(input);
		
		x = featuresModel->forward(input);
		x = x.view({ x.size(0), -1 });
		x = classifierModel->forward(x);
		return x;
	}
	
	torch::Tensor y;
	torch::Tensor x;
	torch::nn::Sequential features;
	torch::nn::Sequential featuresModel;
	torch::nn::Sequential classifierModel;

private:
	int  in_channels;
	void make_model(VGG_CFG cfg, bool nBN = false)
	{
		int v = 0;
		for (int i = 0; i < 21; i++)
		{
			if (vggCfg[cfg][i] == 0)
				break;
			else if (vggCfg[cfg][i] == 2)
				features->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
			else
			{
				v = vggCfg[cfg][i];
				torch::nn::Conv2d conv2d = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, v, 3).padding(1));
				if (nBN)
				{
					features->push_back(conv2d);
					features->push_back(torch::nn::BatchNorm2d(torch::nn::BatchNorm2dOptions(v)));
					features->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
				}
				else
				{
					features->push_back(conv2d);
					features->push_back(torch::nn::ReLU(torch::nn::ReLUOptions(true)));
				}
				in_channels = v;
			}
		}
	}
};


//可以自定义初始化权重,参考如下方式,默认初始化为凯明权重初始化
//void initialize_weights(torch::nn::Module& module)
//{
//	torch::NoGradGuard no_grad;
//	if (auto* linear = module.as<nn::Linear>())
//	{
//		linear->weight.normal_(0.0, 0.02);
//	}
//	else if (auto* conv2d = module.as<nn::Conv2d>())
//	{
//		torch::nn::init::xavier_normal_(conv2d->weight);
//	}
//}