#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void segColor(void);
int createMaskByKmeans(cv::Mat src, cv::Mat& mask);

int main()
{
	//开始计时
	double start = static_cast<double>(getTickCount());
	segColor();
	//结束计时
	double time = ((double)getTickCount() - start) / getTickFrequency();
	//显示时间
	cout << "processing time:" << time / 1000 << "ms" << endl;
	//等待键盘响应，按任意键结束程序
	system("pause");
	return 0;
}


void segColor()
{

	Mat src = imread("../testImages\\movie.jpg");

	Mat mask = Mat::zeros(src.size(), CV_8UC1);
	createMaskByKmeans(src, mask);

	imshow("src", src);
	imshow("mask", mask);

	waitKey(0);

}

int createMaskByKmeans(cv::Mat src, cv::Mat& mask)
{
	if ((mask.type() != CV_8UC1) || (src.size() != mask.size()))
	{
		return 0;
	}

	int width = src.cols;
	int height = src.rows;

	int pixNum = width * height;//像素数量
	int clusterCount = 2;//类的种数，也是判断某个样本为某个类的最少聚类次数
	Mat labels;      //返回的类别标记
	Mat centers;

	//制作kmeans用的数据
	Mat sampleData = src.reshape(3, pixNum);//序列化
	Mat km_data;
	sampleData.convertTo(km_data, CV_32F);//类型转换


	//获取期望精度的迭代最大次数
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	/*
	TermCriteria::TermCriteria(int type, int maxCount, double epsilon)
	参数：
	type –	终止条件类型:
	maxCount –	计算的迭代数或者最大元素数
	epsilon – 当达到要求的精确度或参数的变化范围时，迭代算法停止
	type可选：
	TermCriteria::COUNT //达到最大迭代次数 =TermCriteria::MAX_ITER
	TermCriteria::EPS	//达到精度
	TermCriteria::COUNT + TermCriteria::EPS //以上两种同时作为判定条件
	*/

	//执行kmeans
	kmeans(km_data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);

	//根据聚类结果制作mask
	uchar fg[2] = { 0,255 };
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			mask.at<uchar>(row, col) = fg[labels.at<int>(row * width + col)];
		}
	}

	return 0;
}