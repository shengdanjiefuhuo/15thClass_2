#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void segColor(void);
int createMaskByKmeans(cv::Mat src, cv::Mat& mask);

int main()
{
	//��ʼ��ʱ
	double start = static_cast<double>(getTickCount());
	segColor();
	//������ʱ
	double time = ((double)getTickCount() - start) / getTickFrequency();
	//��ʾʱ��
	cout << "processing time:" << time / 1000 << "ms" << endl;
	//�ȴ�������Ӧ�����������������
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

	int pixNum = width * height;//��������
	int clusterCount = 2;//���������Ҳ���ж�ĳ������Ϊĳ��������پ������
	Mat labels;      //���ص������
	Mat centers;

	//����kmeans�õ�����
	Mat sampleData = src.reshape(3, pixNum);//���л�
	Mat km_data;
	sampleData.convertTo(km_data, CV_32F);//����ת��


	//��ȡ�������ȵĵ���������
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	/*
	TermCriteria::TermCriteria(int type, int maxCount, double epsilon)
	������
	type �C	��ֹ��������:
	maxCount �C	����ĵ������������Ԫ����
	epsilon �C ���ﵽҪ��ľ�ȷ�Ȼ�����ı仯��Χʱ�������㷨ֹͣ
	type��ѡ��
	TermCriteria::COUNT //�ﵽ���������� =TermCriteria::MAX_ITER
	TermCriteria::EPS	//�ﵽ����
	TermCriteria::COUNT + TermCriteria::EPS //��������ͬʱ��Ϊ�ж�����
	*/

	//ִ��kmeans
	kmeans(km_data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);

	//���ݾ���������mask
	uchar fg[2] = { 0,255 };
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			mask.at<uchar>(row, col) = fg[labels.at<int>(row * width + col)];
		}
	}

	return 0;
}