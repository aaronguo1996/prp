#include <iostream>
#include <fstream>
#include <stdio.h>
#include "cxcore.h"
#include "lbp.h"
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
using namespace std;


#define cvQueryHistValue_2D(hist, idx0, idx1)   cvGetReal2D((hist)->bins, (idx0), (idx1))
#define GLCM_DIS 3  //�Ҷȹ��������ͳ�ƾ���
#define GLCM_CLASS 16 //����Ҷȹ��������ͼ��Ҷ�ֵ�ȼ���
#define GLCM_ANGLE_HORIZATION 0  //ˮƽ
#define GLCM_ANGLE_VERTICAL   1	 //��ֱ
#define GLCM_ANGLE_DIGONAL    2  //�Խ�
#define HBINS 16
#define SBINS 8
#define VBINS 8
#define IMGW 256
#define IMGH 384
#define MATSIZE (HBINS*SBINS)//(IMGW*IMGH)//12//(HBINS*SBINS)
#define GLCMSIZE 12

Mat hsvArray;
Mat glcmArray;
Mat data_mat;
Mat res_mat;
Mat fusion_mat;

int glcm[GLCM_CLASS * GLCM_CLASS];
int histImage[IMGW*IMGH];
char *name = new char[64];
int nLine = 0;
int descSize = 0;
vector<int>res_color;
vector<int>res_glcm;
vector<int>res_hog;
vector<int>res_final;

double minValue[12];
double maxValue[12];

void hsv(string name, int cnt)
{
	IplImage * src = cvLoadImage(name.c_str(), 1);
	IplImage* hsv = cvCreateImage(cvGetSize(src), 8, 3);
	IplImage* h_plane = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* s_plane = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* v_plane = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* planes[] = { h_plane, s_plane };

	/** H ��������Ϊ16���ȼ���S��������Ϊ8���ȼ�*/
	int h_bins = 16, s_bins = 8;
	int hist_size[] = { h_bins, s_bins };

	/** H �����ı仯��Χ*/
	float h_ranges[] = { 0, 180 };//[TODO]
	/** S �����ı仯��Χ*/
	float s_ranges[] = { 0, 255 };
	float* ranges[] = { h_ranges, s_ranges };

	/** ����ͼ��ת����HSV��ɫ�ռ�*/
	cvCvtColor(src, hsv, CV_BGR2HSV);//[TODO]
	cvSplit(hsv, h_plane, s_plane, v_plane, 0);

	/** ����ֱ��ͼ����ά, ÿ��ά���Ͼ���*/
	CvHistogram * hist = cvCreateHist(2, hist_size, CV_HIST_ARRAY, ranges, 1);

	/** ����H,S����ƽ������ͳ��ֱ��ͼ*/
	cvCalcHist(planes, hist, 0, 0);

	/** ��ȡֱ��ͼͳ�Ƶ����ֵ�����ڶ�̬��ʾֱ��ͼ*/
	float max_value;
	cvGetMinMaxHistValue(hist, 0, &max_value, 0, 0);

	for (int h = 0; h < h_bins; h++)
	{
		for (int s = 0; s < s_bins; s++)
		{
			int i = h*s_bins + s;

			/** ���ֱ��ͼ�е�ͳ�ƴ�����������ʾ��ͼ���еĸ߶�*/
			float bin_val = cvQueryHistValue_2D(hist, h, s);
			//int intensity = cvRound(bin_val*height / max_value);
			hsvArray.at<float>(cnt, i) = bin_val/max_value;//intensity;
			//fusion_mat.at<float>(cnt, i+GLCMSIZE+descSize) = bin_val / max_value;
		}
	}
	cvReleaseImage(&src);
	cvReleaseImage(&hsv);
	cvReleaseImage(&h_plane);
	cvReleaseImage(&s_plane);
	cvReleaseImage(&v_plane);
}

int calGLCM(string name, int angleDirection, int cnt)
{
	IplImage *src = cvLoadImage(name.c_str(), 1);
	IplImage* bWavelet = cvCreateImage(cvGetSize(src), 8, 1);
	cvCvtColor(src, bWavelet, CV_BGR2GRAY);

	int i, j;
	int width, height;

	if (NULL == bWavelet)
		return 1;

	width = bWavelet->width;
	height = bWavelet->height;

	//memset(glcm, 0, GLCM_CLASS * GLCM_CLASS);
	//memset(histImage, 0, IMGW*IMGH);
	if (NULL == glcm || NULL == histImage)
		return 2;

	// ---��GLCM_CLASS���ȼ�
	uchar *data = (uchar*)bWavelet->imageData;
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			histImage[i * width + j] = (int)(data[bWavelet->widthStep * i + j] * GLCM_CLASS / 256);
		}
	}

	//��ʼ����������
	for (i = 0; i < GLCM_CLASS; i++)
		for (j = 0; j < GLCM_CLASS; j++)
			glcm[i * GLCM_CLASS + j] = 0;

	//����Ҷȹ�������
	int w, k, l;
	//ˮƽ����
	if (angleDirection == GLCM_ANGLE_HORIZATION)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				l = histImage[i * width + j];
				if (j + GLCM_DIS >= 0 && j + GLCM_DIS < width)
				{
					k = histImage[i * width + j + GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
				if (j - GLCM_DIS >= 0 && j - GLCM_DIS < width)
				{
					k = histImage[i * width + j - GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}
	//��ֱ����
	else if (angleDirection == GLCM_ANGLE_VERTICAL)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				l = histImage[i * width + j];
				if (i + GLCM_DIS >= 0 && i + GLCM_DIS < height)
				{
					k = histImage[(i + GLCM_DIS) * width + j];
					glcm[l * GLCM_CLASS + k]++;
				}
				if (i - GLCM_DIS >= 0 && i - GLCM_DIS < height)
				{
					k = histImage[(i - GLCM_DIS) * width + j];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}
	//�ԽǷ���
	else if (angleDirection == GLCM_ANGLE_DIGONAL)
	{
		for (i = 0; i < height; i++)
		{
			for (j = 0; j < width; j++)
			{
				l = histImage[i * width + j];

				if (j + GLCM_DIS >= 0 && j + GLCM_DIS < width && i + GLCM_DIS >= 0 && i + GLCM_DIS < height)
				{
					k = histImage[(i + GLCM_DIS) * width + j + GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
				if (j - GLCM_DIS >= 0 && j - GLCM_DIS < width && i - GLCM_DIS >= 0 && i - GLCM_DIS < height)
				{
					k = histImage[(i - GLCM_DIS) * width + j - GLCM_DIS];
					glcm[l * GLCM_CLASS + k]++;
				}
			}
		}
	}

	//��������ֵ
	double entropy = 0, energy = 0, contrast = 0, homogenity = 0;
	for (i = 0; i < GLCM_CLASS; i++)
	{
		for (j = 0; j < GLCM_CLASS; j++)
		{
			//��
			if (glcm[i * GLCM_CLASS + j] > 0)
				entropy -= glcm[i * GLCM_CLASS + j] * log10(double(glcm[i * GLCM_CLASS + j]));
			//����
			energy += glcm[i * GLCM_CLASS + j] * glcm[i * GLCM_CLASS + j];
			//�Աȶ�
			contrast += (i - j) * (i - j) * glcm[i * GLCM_CLASS + j];
			//һ����
			homogenity += 1.0 / (1 + (i - j) * (i - j)) * glcm[i * GLCM_CLASS + j];
		}
	}
	//��������ֵ
	i = 4*angleDirection;
	glcmArray.at<float>(cnt,i++) = entropy;
	glcmArray.at<float>(cnt,i++) = energy;
	glcmArray.at<float>(cnt,i++) = contrast;
	glcmArray.at<float>(cnt,i++) = homogenity;
	//fusion_mat.at<float>(cnt, i++) = entropy;
	//fusion_mat.at<float>(cnt, i++) = energy;
	//fusion_mat.at<float>(cnt, i++) = contrast;//delete[] glcm;
	//fusion_mat.at<float>(cnt, i++) = homogenity;//delete[] histImage;
	return 0;
}

int hog(string name, int i)
{
	int ImgWidht = 120;
	int ImgHeight = 120;

	Mat src;
	Mat trainImg = Mat::zeros(ImgHeight, ImgWidht, CV_8UC3);//��Ҫ������ͼƬ  
	
	src = imread(name.c_str(), 1);
	//cout << "HOG: processing " << name.c_str() << endl;
	resize(src, trainImg, cv::Size(ImgWidht, ImgHeight), 0, 0, INTER_CUBIC);
	HOGDescriptor *hog = new HOGDescriptor(cvSize(ImgWidht, ImgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);     
	
	vector<float>descriptors;//�������     
	hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //���ü��㺯����ʼ����
	if (i == 0)
	{
		//descSize = descriptors.size();
		data_mat = Mat::zeros(nLine, descriptors.size(), CV_32FC1); //��������ͼƬ��С���з���ռ� 
		//fusion_mat = Mat::zeros(nLine, descriptors.size() + MATSIZE + GLCMSIZE, CV_32FC1);
	}
	int n = 0;
	for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
	{
		data_mat.at<float>(i, n) = *iter;
		//fusion_mat.at<float>(i, n) = *iter;
		n++;
	}
	//cout << "HOG: end processing " << name.c_str() << endl;
	delete hog;
	
	return 0;
}


int testColorHist(Mat data, Mat res)
{
	/////////////START SVM TRAINNING//////////////////
	CvSVM svm;
	CvSVMParams param;
	CvTermCriteria criteria;

	criteria = cvTermCriteria(CV_TERMCRIT_ITER , 10000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.5, 1.0, 1, 0.5, 1, NULL, criteria);//for color hist
	svm.train(hsvArray, res_mat, Mat(), Mat(), param);

	string buf;
	vector<string> img_tst_path;
	ifstream img_tst("SVM_TEST.txt");
	while (img_tst)
	{
		if (getline(img_tst, buf))
		{
			img_tst_path.push_back(buf);
		}
	}
	img_tst.close();

	char line[512];
	ofstream predict_txt("SVM_PREDICT_COLOR.txt");
	for (int i = 0; i< img_tst_path.size(); i++)
	{
		cout << "The Detection Result:" << endl;
		//hsvArray = Mat::zeros(img_tst_path.size(), MATSIZE, CV_32FC1);
		Mat m = Mat::zeros(1, MATSIZE, CV_32FC1);

		hsv(img_tst_path[i].c_str(), i);
		for (int j = 0; j < HBINS*SBINS; j++)
		{
			m.at<float>(0,j) = hsvArray.at<float>(i,j);
		}

		int ret = 0;
		ret = svm.predict(m);
		res_color.push_back(ret);
		std::sprintf(line, "%s\t%d\n", img_tst_path[i].c_str(), ret);
		printf("%s %d\n", img_tst_path[i].c_str(), ret);
		predict_txt << line;
	}

	return 0;
}

int testGlcm(Mat data, Mat res)
{
	/////////////START SVM TRAINNING//////////////////
	CvSVM svm;
	CvSVMParams param;
	CvTermCriteria criteria;

	criteria = cvTermCriteria(CV_TERMCRIT_ITER, 10000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.75, 0.001, 100.0, 0.75, 0.5, NULL, criteria);//for glcm
	svm.train(glcmArray, res_mat, Mat(), Mat(), param);

	string buf;
	vector<string> img_tst_path;
	ifstream img_tst("SVM_TEST.txt");
	while (img_tst)
	{
		if (getline(img_tst, buf))
		{
			img_tst_path.push_back(buf);
		}
	}
	img_tst.close();

	char line[512];
	ofstream predict_txt("SVM_PREDICT_GLCM.txt");
	for (int i = 0; i< img_tst_path.size(); i++)
	{
		cout << "The Detection Result:" << endl;
		Mat m_dst = Mat::zeros(1, GLCMSIZE, CV_32FC1);

		//�Խ��߷���
		calGLCM(img_tst_path[i].c_str(), GLCM_ANGLE_DIGONAL, i);
		calGLCM(img_tst_path[i].c_str(), GLCM_ANGLE_HORIZATION, i);
		calGLCM(img_tst_path[i].c_str(), GLCM_ANGLE_VERTICAL, i);

		for (int j = 0; j < 12; j++)
			m_dst.at<float>(0, j) = (glcmArray.at<float>(i, j) - minValue[j]) / (maxValue[j] - minValue[j]);

		int ret = 0;
		ret = svm.predict(m_dst);
		res_glcm.push_back(ret);
		std::sprintf(line, "%s\t%d\n", img_tst_path[i].c_str(), ret);
		printf("%s %d\n", img_tst_path[i].c_str(), ret);
		predict_txt << line;
	}
	return 0;
}

int testHOG(Mat data, Mat res)
{
	CvSVM svm;
	CvSVMParams param;
	CvTermCriteria criteria;
	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::LINEAR, 10.0, 0.1, 0.09, 100.0, 0.5, 1.0, NULL, criteria);//for hog
	svm.train(data, res, Mat(), Mat(), param);
	int ImgWidth = 120;
	int ImgHeight = 120;
	string buf;
	vector<string> img_tst_path;
	ifstream img_tst("SVM_TEST.txt");
	while (img_tst)
	{
		if (getline(img_tst, buf))
		{
			img_tst_path.push_back(buf);
		}
	}
	img_tst.close();

	Mat test;
	Mat trainImg = Mat::zeros(ImgHeight, ImgWidth, CV_8UC3);//��Ҫ������ͼƬ  
	char line[512];
	ofstream predict_txt("SVM_PREDICT_HOG.txt");
	for (string::size_type j = 0; j != img_tst_path.size(); j++)
	{
		test = imread(img_tst_path[j].c_str(), 1);//����ͼ��   
		resize(test, trainImg, cv::Size(ImgWidth, ImgHeight), 0, 0, INTER_CUBIC);//Ҫ���ͬ���Ĵ�С�ſ��Լ�⵽       
		HOGDescriptor *hog = new HOGDescriptor(cvSize(ImgWidth, ImgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
		vector<float>descriptors;//�������     
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //���ü��㺯����ʼ���� 
		cout << "The Detection Result:" << endl;

		Mat SVMtrainMat = Mat::zeros(1, descriptors.size(), CV_32FC1);
		int n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			SVMtrainMat.at<float>(0, n) = *iter;
			n++;
		}

		int ret = svm.predict(SVMtrainMat);
		res_hog.push_back(ret);
		std::sprintf(line, "%s\t%d\n", img_tst_path[j].c_str(), ret);
		printf("%s %d\n", img_tst_path[j].c_str(), ret);
		predict_txt << line;
		delete hog;
	}
	predict_txt.close();
	return 0;
}

int testFusion(Mat data,Mat res)
{
	CvSVM svm;
	CvSVMParams param;
	CvTermCriteria criteria;
	criteria = cvTermCriteria(CV_TERMCRIT_ITER, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 0.09, 0.1, 100.0, 0.5, 1.0, NULL, criteria);//for hog
	svm.train(data, res, Mat(), Mat(), param);
	int ImgWidth = 120;
	int ImgHeight = 120;
	string buf;
	vector<string> img_tst_path;
	ifstream img_tst("SVM_TEST.txt");
	while (img_tst)
	{
		if (getline(img_tst, buf))
		{
			img_tst_path.push_back(buf);
		}
	}
	img_tst.close();

	Mat test;
	Mat trainImg = Mat::zeros(ImgHeight, ImgWidth, CV_8UC3);//��Ҫ������ͼƬ  
	char line[512];
	ofstream predict_txt("SVM_PREDICT_FUSION.txt");
	for (string::size_type j = 0; j != img_tst_path.size(); j++)
	{
		test = imread(img_tst_path[j].c_str(), 1);//����ͼ��   
		resize(test, trainImg, cv::Size(ImgWidth, ImgHeight), 0, 0, INTER_CUBIC);//Ҫ���ͬ���Ĵ�С�ſ��Լ�⵽       
		HOGDescriptor *hog = new HOGDescriptor(cvSize(ImgWidth, ImgHeight), cvSize(16, 16), cvSize(8, 8), cvSize(8, 8), 9);
		vector<float>descriptors;//�������     
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //���ü��㺯����ʼ���� 
		cout << "The Detection Result:" << endl;

		Mat SVMtrainMat = Mat::zeros(1, GLCMSIZE+MATSIZE+descriptors.size(), CV_32FC1);
		int n = 0;
		for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
		{
			SVMtrainMat.at<float>(0, n) = *iter;
			n++;
		}
		//����GLCM
		calGLCM(img_tst_path[j].c_str(), GLCM_ANGLE_DIGONAL, j);
		calGLCM(img_tst_path[j].c_str(), GLCM_ANGLE_HORIZATION, j);
		calGLCM(img_tst_path[j].c_str(), GLCM_ANGLE_VERTICAL, j);
		for (int k = 0; k < 12; k++)
			SVMtrainMat.at<float>(0, k + descSize) = (fusion_mat.at<float>(j, k + descSize) - minValue[k]) / (maxValue[k] - minValue[k]);

		//������ɫֱ��ͼ
		hsv(img_tst_path[j].c_str(), j);
		for (int k = 0; k < MATSIZE; k++)
		{
			SVMtrainMat.at<float>(0, GLCMSIZE + descSize + k) = fusion_mat.at<float>(j, GLCMSIZE + descSize + k);
		}
		int ret = svm.predict(SVMtrainMat);
		std::sprintf(line, "%s\t%d\n", img_tst_path[j].c_str(), ret);
		printf("%s %d\n", img_tst_path[j].c_str(), ret);
		predict_txt << line;
		delete hog;
	}
	predict_txt.close();
	return 0;
}

int voteForResults(int voter1,int voter2,int voter3)
{
	if (voter1 == voter2 || voter1 == voter3)
	{
		return voter1;
	}
	else if (voter2 == voter3 || voter2 == voter1)
	{
		return voter2;
	}
	else if (voter3 == voter1 || voter3 == voter2)
	{
		return voter3;
	}
	else if (voter1 != voter2 && voter1 != voter3 && voter2 != voter3)
	{
		return voter1;
	}
}

void drawColorHist()
{
	IplImage * src = cvLoadImage("image.orig/720.jpg", 1);
	IplImage* hsv = cvCreateImage(cvGetSize(src), 8, 3);
	IplImage* h_plane = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* s_plane = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* v_plane = cvCreateImage(cvGetSize(src), 8, 1);
	IplImage* planes[] = { h_plane, s_plane };

	/** H ��������Ϊ16���ȼ���S��������Ϊ8���ȼ�*/
	int h_bins = 16, s_bins = 8;
	int hist_size[] = { h_bins, s_bins };
	/** H �����ı仯��Χ*/
	float h_ranges[] = { 0, 180 };
	/** S �����ı仯��Χ*/
	float s_ranges[] = { 0, 255 };
	float* ranges[] = { h_ranges, s_ranges };

	/** ����ͼ��ת����HSV��ɫ�ռ�*/
	cvCvtColor(src, hsv, CV_BGR2HSV);
	cvCvtPixToPlane(hsv, h_plane, s_plane, v_plane, 0);
	/** ����ֱ��ͼ����ά, ÿ��ά���Ͼ���*/
	CvHistogram * hist = cvCreateHist(2, hist_size, CV_HIST_ARRAY, ranges, 1);
	/** ����H,S����ƽ������ͳ��ֱ��ͼ*/
	cvCalcHist(planes, hist, 0, 0);

	/** ��ȡֱ��ͼͳ�Ƶ����ֵ�����ڶ�̬��ʾֱ��ͼ*/
	float max_value;
	cvGetMinMaxHistValue(hist, 0, &max_value, 0, 0);

	/** ����ֱ��ͼ��ʾͼ��*/
	int height = 240;
	int width = (h_bins*s_bins * 6);
	IplImage* hist_img = cvCreateImage(cvSize(width, height), 8, 3);
	cvZero(hist_img);

	/** ��������HSV��RGB��ɫת������ʱ��λͼ��*/
	IplImage * hsv_color = cvCreateImage(cvSize(1, 1), 8, 3);
	IplImage * rgb_color = cvCreateImage(cvSize(1, 1), 8, 3);
	int bin_w = width / (h_bins * s_bins);
	for (int h = 0; h < h_bins; h++)
	{
		for (int s = 0; s < s_bins; s++)
		{
			int i = h*s_bins + s;
			/** ���ֱ��ͼ�е�ͳ�ƴ�����������ʾ��ͼ���еĸ߶�*/
			float bin_val = cvQueryHistValue_2D(hist, h, s);
			int intensity = cvRound(bin_val*height / max_value);
			/** ��õ�ǰֱ��ͼ�������ɫ��ת����RGB���ڻ���*/
			cvSet2D(hsv_color, 0, 0, cvScalar(h*180.f / h_bins, s*255.f / s_bins, 255, 0));
			cvCvtColor(hsv_color, rgb_color, CV_HSV2BGR);
			CvScalar color = cvGet2D(rgb_color, 0, 0);

			cvRectangle(hist_img, cvPoint(i*bin_w, height),
				cvPoint((i + 1)*bin_w, height - intensity),
				color, -1, 8, 0);
		}
	}
	cvNamedWindow("H-S Histogram", 1);
	cvShowImage("H-S Histogram", hist_img);

	cvWaitKey(0);
}

void cal()
{
	vector<string> img_path;
	vector<int> img_catg;
	string buf;
	ifstream svm_data("SVM_DATA.txt");
	unsigned long n;

	while (svm_data)
	{
		if (getline(svm_data, buf))
		{
			int pos = buf.find_first_of('\t');
			nLine++;
			img_catg.push_back(atoi(buf.substr(pos + 1).c_str()));
			img_path.push_back(buf.substr(0, pos));//ͼ��·�� 
		}
	}
	svm_data.close();//�ر��ļ�

	//���;���,�洢ÿ�����������ͱ�־  
	res_mat = Mat::zeros(nLine, 1, CV_32FC1);
	hsvArray = Mat::zeros(nLine, MATSIZE, CV_32FC1);
	glcmArray = Mat::zeros(nLine, GLCMSIZE, CV_32FC1);

	//������ȡ
	for (int i = 0; i < img_path.size(); i++)
	{
		cout << "Processing " << img_path[i] << "..." << img_catg[i]<< endl;
		hog(img_path[i].c_str(), i);
		hsv(img_path[i], i);
		calGLCM(img_path[i].c_str(), GLCM_ANGLE_DIGONAL, i);
		calGLCM(img_path[i].c_str(), GLCM_ANGLE_HORIZATION, i);
		calGLCM(img_path[i].c_str(), GLCM_ANGLE_VERTICAL, i);
		res_mat.at<float>(i) = img_catg[i];
		cout << "End Processing " << img_path[i] << "..." << endl;
		cout << "***************************************" << endl;
	}

	//�ҳ�GLCM���е���ֵ
	for (int i = 0; i < 12; i++)
	{
		minValue[i] = glcmArray.at<float>(0, i);
		maxValue[i] = glcmArray.at<float>(0, i);
	}
	for (int i = 1; i < nLine; i++)
	{
		for (int j = 0; j < 12; j++)
		{
			if (glcmArray.at<float>(i, j) < minValue[j])
				minValue[j] = glcmArray.at<float>(i, j);
			if (glcmArray.at<float>(i, j) > maxValue[j])
				maxValue[j] = glcmArray.at<float>(i, j);
		}
	}
	//GLCM���ݹ�һ��
	for (int i = 1; i < nLine; i++)
	{
		for (int j = 0; j < 12; j++)
		{
			glcmArray.at<float>(i, j) = (glcmArray.at<float>(i, j) - minValue[j]) / (maxValue[j] - minValue[j]);
		}
	}

	//ѵ��SVM���������н��Ԥ��
	testColorHist(hsvArray, res_mat);
	//testGlcm(glcmArray, res_mat);
	//testHOG(data_mat, res_mat);
	//testFusion(fusion_mat,res_mat);

	ofstream ofs("SVM_PREDICT_FUSION.txt");
	for (int i = 0; i < 100; i++)
	{
		res_final.push_back(/*voteForResults(res_hog[i], res_color[i], res_glcm[i])*/res_color[i]);
		ofs << res_final[i] <<endl;
	}
}

int main()
{
	cal();
	return 0;
}