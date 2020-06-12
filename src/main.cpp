// Offline_detection_v2.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//需要用户修改的参数：
//326行：文件名
//344行：fileSpeed; 球速，参考文件名
//345行：fileAngle; 开始的角度
//346行：Upup;    角度增大：1，角度减小：0
//348行：jumpTo； 可以选择从第几帧开始
//
//功能说明：第一帧需要在左右目图像中分别框出发球位置
//按Esc键 下一帧
//如果还没发球就误检测到球，按t键复位
//如果对检测结果不满意，按p键 可在左右目图像中分别定位球的位置（选择方法：单击鼠标左键，然后按空格；如果对结果满意，按Esc继续，如果不满意，按其余任意键重新标定）除非误差很大，一般不做此操作
//如果程序意外退出，或忘记对上一帧做某些操作，可记下当前帧数，关闭程序，修改jumpTo变量，重新开始
//

#include "pch.h"
#include "opencv2/opencv.hpp"
#include<iostream>
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

using namespace std;
using namespace cv;

constexpr auto m = 0.005;;
constexpr auto g = 9.8;
constexpr auto b = 0.001155;

const float resize_fact = 1;
const int imageWidth = 1280;                      //【可能需要修改的程序参数1】：单目图像的宽度
const int imageHeight = 960;
const float predict_box_width = 160, predict_box_height = 160;

float pixel_length = 0.00375;
float centerL_x = pixel_length * imageWidth * resize_fact / 2, centerL_y = pixel_length * imageHeight * resize_fact / 2, centerR_x = centerL_x, centerR_y = centerL_y;
float baseline = 190;
float focal_length = 2;
float target_z = 0, target_xp = 0, target_xpp = 0, target_y = 0;
float xl = 0, xr = 0, yl = 0, yr = 0;
float next_location[3] = { 0 };
int lost_countL = 0, lost_countR = 0;
int last_is_predictedL = 0, last_is_predictedR = 0;
int isDetected = 0;

// Extended Kalman coefficients
float K_large[6] = { 0.31508, 0.14199, 0.31597, 0.15356, 0.30786, 0.040882 };
float K_small[6] = { 0.31508/5, 0.14199/5, 0.31597/5, 0.15356/5, 0.30786/5, 0.040882/5 };

/*左目相机标定参数------------------------
fc_left_x   0            cc_left_x
0           fc_left_y    cc_left_y
0           0            1
-----------------------------------------*/

Mat cameraMatrixL = (Mat_<double>(3, 3) << 917.69097, 0, 677.95004,
	0, 918.11396, 498.02874,
	0, 0, 1);


Mat distCoeffL = (Mat_<double>(5, 1) << 0.03362, -0.01240, -0.00197, -0.00262, 0.00000);
//[kc_left_01,  kc_left_02,  kc_left_03,  kc_left_04,   kc_left_05]


/*右目相机标定参数------------------------
fc_right_x   0              cc_right_x
0            fc_right_y     cc_right_y
0            0              1
-----------------------------------------*/
Mat cameraMatrixR = (Mat_<double>(3, 3) << 914.86627, 0, 631.49194,
	0, 914.56919, 452.46808,
	0, 0, 1);


Mat distCoeffR = (Mat_<double>(5, 1) << 0.05988, -0.06385, -0.00799, 0.00161, 0.00000);
//[kc_right_01,  kc_right_02,  kc_right_03,  kc_right_04,   kc_right_05]


Mat T = (Mat_<double>(3, 1) << -189.50993, -0.42497, 3.96392);    //T平移向量
							 //[T_01,        T_02,       T_03]

Mat rec = (Mat_<double>(3, 1) << -0.03862, 0.02930, -0.00266);   //rec旋转向量
							  //[rec_01,     rec_02,     rec_03]

//########--双目标定参数填写完毕-----------------------------------------------------------------------


Mat R;
Mat MoveDetect(Mat frame1, Mat frame2, Rect2d roi, int isRight)
{
	Mat result = frame2.clone();

	double T0 = (double)getTickCount();//开始时间
	double t_passed = 0.0;
	Mat sub;
	sub = (frame2 - frame1);
	sub = max(0, sub);

	//ONLY DETECT ROI
	sub = sub(roi);
	//cout << "Sub image size: " << sub.size() << endl;

	threshold(sub, sub, 7, 255, CV_THRESH_BINARY);
	imshow("threshold", sub);

	Mat element_1 = getStructuringElement(MORPH_RECT, Size(3, 3));
	//Mat kernel(2, 3, CV_8UC1);
	erode(sub, sub, element_1);
	imshow("erode", sub);


	Mat element_2 = getStructuringElement(MORPH_RECT, Size(7, 7));
	dilate(sub, sub, element_2);//先腐蚀再膨胀 开运算 除去毛刺以及孤立小点

	threshold(sub, sub, 250, 255, CV_THRESH_BINARY);
	imshow("dilate", sub);

	vector<vector<Point>> contours;//双重向量 每一组点集就是一个轮廓，有多少轮廓，contours就有多少元素
	vector<Vec4i> hierarcy;//hierarchy是一个向量，向量内每个元素都是一个包含4个int型的数组.
						   //hierarchy内每个元素的4个int型变量是hierarchy[i][0] ~ hierarchy[i][3]，分别表示当前轮廓 i 的后一个轮廓、前一个轮廓、父轮廓和内嵌轮廓的编号索引。
	//画椭圆及中心

	findContours(sub, contours, hierarcy, RETR_TREE, CV_CHAIN_APPROX_NONE);//只检测最外围轮廓，包含在外围轮廓内的内围轮廓被忽略；//保存物体边界上所有连续的轮廓点到contours向量内；

	//画预测矩形框
	rectangle(result, cvPoint(roi.x, roi.y), cvPoint(roi.x + roi.width, roi.y + roi.height), cvScalar(0, 0, 255), 3, 4, 0);
	isDetected = contours.size();

	//cout << "num of badmintons =" << contours.size() << endl;
	//cout << "roi_x: " << roi.x << endl; 
	//cout << "roi_y: " << roi.y << endl;
	float true_x = 0, true_y = 0;
	float distance = 0;
	float min_distance = 90000;
	vector<RotatedRect> box(contours.size());
	if (contours.size() == 0)
	{
		if (isRight == 0)
		{
			lost_countL++;
			//xl = (roi.x + roi.width / 2) * pixel_length - centerL_x;
			//yl = centerL_y - (roi.y + roi.height / 2) * pixel_length;
			last_is_predictedL = 1;
			//cout << "Left lost!" << endl;
		}
		else if (isRight == 1)
		{
			lost_countR++;
			//xr = (roi.x + roi.width / 2) * pixel_length - centerR_x;
			//yr = centerR_y - (roi.y + roi.height / 2) * pixel_length;
			last_is_predictedR = 1;
			//cout << "Right lost!" << endl;
		}
		cv::circle(result, Point((roi.x + roi.width / 2), (roi.y + roi.height / 2)), 3, Scalar(0, 0, 255), -1, 8);
		//cout << "THIS FRAME IS PREDICTED!" << endl;
	}
	else
	{
		for (int i = 0; i < contours.size(); i++)
		{
			if (contours[i].size() >= 5)
			{
				box[i] = fitEllipse(Mat(contours[i]));// ******************************怎么把这个向量输出出来
				distance = sqrt(pow((box[i].center.x - (roi.width / 2)), 2) + pow((box[i].center.y - (roi.height / 2)), 2));
				//cout << "center" << i << ": " << box[i].center << endl;
				if (1 /*box[i].size.width > 7 && box[i].size.height > 7*/)
				{
					//if (box[i].center.x > max_x)
						//max_x = box[i].center.x;
					if (distance < min_distance)
					{
						min_distance = distance;
						true_x = box[i].center.x;
						true_y = box[i].center.y;
						true_x += roi.x;
						true_y += roi.y;
					}

					//cout << "center_roi" << i << ": " << box[i].center << endl;
					//cout << "size: " << min_x << "  " << min_x << endl;
					//计算球的三维坐标
					if (isRight == 0)
					{
						if (last_is_predictedL == 1)
							last_is_predictedL = 0;
						else
							lost_countL = 0;
						xl = true_x * pixel_length - centerL_x;
						yl = centerL_y - true_y * pixel_length;
						//cout << "Left location in image: " << xl << ", " << yl << endl;
					}
					else if (isRight == 1)
					{
						if (last_is_predictedR == 1)
							last_is_predictedR = 0;
						else
							lost_countR = 0;
						xr = true_x * pixel_length - centerL_x;
						yr = centerL_y - true_y * pixel_length;
						//cout << "Right location in image: " << xr << ", " << yr << endl;
					}

				}
			}
		}
		cv::circle(result, Point(true_x, true_y), 3, Scalar(0, 0, 255), -1, 8);
	}
	t_passed = (double)getTickCount() - T0;//代码运行时间=结束时间-开始时间
	printf("DRAW===================------execution time = %gms\n", t_passed*1000. / getTickFrequency());//转换时间单位并输出代码运行时间

	return result;
}

class BoxExtractor {
public:
	Rect2d extract(Mat img);
	Rect2d extract(const std::string& windowName, Mat img, bool showCrossair = true);

	struct handlerT {
		bool isDrawing;
		Rect2d box;
		Mat image;

		// initializer list
		handlerT() : isDrawing(false) {};
	}params;

private:
	static void mouseHandler(int event, int x, int y, int flags, void *param);
	void opencv_mouse_callback(int event, int x, int y, int, void *param);
};

void predict_next_location(float pre_x, float pre_y, float pre_z, float cur_x, float cur_y, float cur_z, float sample_period)
{
	static float new_pre[3] = { 0 };
	static float new_cur[3] = { 0 };
	static float new_next[3] = { 0 };
	static float vel[3] = { 0 };
	static float acc[3] = { 0 };
	static float v_total = 0;
	static float swing_angle = 0;
	static float elevation_angle = 0;

	pre_x = pre_x / 1000;
	pre_y = pre_y / 1000;
	pre_z = pre_z / 1000;
	cur_x = cur_x / 1000;
	cur_y = cur_y / 1000;
	cur_z = cur_z / 1000;

	new_cur[0] = cur_x - pre_x;
	new_cur[1] = cur_z - pre_z;
	new_cur[2] = cur_y - pre_y;
	cout << "new_cur: (" << new_cur[0] << "," << new_cur[1] << "," << new_cur[2] << ")" << endl;

	swing_angle = atan(new_cur[0] / new_cur[1]);
	v_total = sqrt(pow(new_cur[0], 2) + pow(new_cur[1], 2) + pow(new_cur[2], 2)) / sample_period;
	elevation_angle = atan(new_cur[2] / sqrt(pow(new_cur[0], 2) + pow(new_cur[1], 2)));

	cout << "swing, v, ele : (" << swing_angle << "," << v_total << "," << elevation_angle << ")" << endl;

	vel[0] = v_total * cos(elevation_angle) * sin(swing_angle);
	vel[1] = v_total * cos(elevation_angle) * cos(swing_angle);
	vel[2] = v_total * sin(elevation_angle);

	cout << "vel: (" << vel[0] << "," << vel[1] << "," << vel[2] << ")" << endl;

	acc[0] = -b / m * v_total * vel[0];
	acc[1] = -b / m * v_total * vel[1];
	acc[2] = -g - b / m * v_total * vel[2];
	cout << "acc : (" << acc[0] << "," << acc[1] << "," << acc[2] << ")" << endl;

	new_next[0] = acc[0] * pow(sample_period, 2) + 2 * new_cur[0] - new_pre[0];
	new_next[1] = acc[1] * pow(sample_period, 2) + 2 * new_cur[1] - new_pre[1];
	new_next[2] = acc[2] * pow(sample_period, 2) + 2 * new_cur[2] - new_pre[2];



	next_location[0] = (new_next[0] + pre_x) * 1000;
	next_location[1] = (new_next[2] + pre_y) * 1000;
	next_location[2] = (new_next[1] + pre_z) * 1000;

	cout << "predicted location: (" << next_location[0] << "," << next_location[1] << "," << next_location[2] << ")" << endl;


}

void main()
{
	float pre_location[3] = { 0 };
	float current_location[3] = { 0 };
	float next_xl = 0, next_xr = 0, next_yl = 0, next_yr = 0;
	Size dstSize, dstSize_double;
	Size imageSize;
	dstSize.width = imageWidth * resize_fact;
	dstSize.height = imageHeight * resize_fact;
	dstSize_double.width = imageWidth * resize_fact * 2;
	dstSize_double.height = dstSize.height;
	imageSize.width = imageWidth;
	imageSize.height = imageHeight;

	// ROI selector
	BoxExtractor box;
	Mat frame;
	Mat grayImageL;// = Mat::zeros(imageHeight, imageWidth * 2, frame.type());
	Mat grayImageR;// = Mat::zeros(imageHeight, imageWidth * 2, frame.type());
	Mat rectifyImageL;// = Mat::zeros(imageHeight, imageWidth * 2, frame.type()),
	Mat rectifyImageR;// = Mat::zeros(imageHeight, imageWidth * 2, frame.type());

	Rect validROIL;                                   //图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
	Rect validROIR;
	Mat mapLx, mapLy, mapRx, mapRy;                   //映射表  
	Mat Rl, Rr, Pl, Pr, Q;                            //校正旋转矩阵R，投影矩阵P, 重投影矩阵Q
	//Mat xyz;                                          //用于存放每个像素点距离相机镜头的三维坐标


	ifstream fin("Test.txt");
	if (fin) {
		cout << "Please rename exist file! Exit." << endl;
		return;
	}

	ofstream OutFile("Test.txt");
	OutFile << "1: Speed, Angle.  0: Trajectory." << endl;
	cout << "1: Speed, Angle.  0: Trajectory." << endl;

	//=================================图像矫正相关操作==================================
	Rodrigues(rec, R);                                   //Rodrigues变换
	stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
		0, imageSize, &validROIL, &validROIR);
	initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
	initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
	//=================================图像矫正相关操作==================================

	VideoCapture cap("HIGH-S130.mp4");

	if (!cap.isOpened())
	{
		cout << "can't open video." << endl;
		return;
	}

	Mat resultL, resultR;
	Mat backgroundL, backgroundR;
	//Mat subImage;
	Rect2d roi_launchL, roi_launchR, roi_nextL, roi_nextR;
	Rect2d predict_roi_L, predict_roi_R;
	int count = 0;

	int isLaunch = 0, ball_in_air = 0, in_air_count = 0;
	int target_vel = 0;
	int keykey = 0;
	int fileSpeed = 70;
	int fileAngle = 22;
	int Upup = 0;
	int isLastPoint = 0;
	int jumpTo = 1700;
	if (jumpTo != 0)
	{
		int i = 0;
		while (i < jumpTo)
		{
			cap >> frame;
			i++;

		}
	}

	//State of the shuttlecock
	float m_neg[6] = { 0 }, m_pos[6] = { 0 };
	float T = 0.033, btm = 0.0015*0.033/0.0049, gT = 9.8*0.033;

	while (1)
	{
		Mat gray_frame0 = Mat::zeros(640, 360, frame.type());//全0矩阵;
		Mat gray_frame;// = Mat::zeros(imageHeight, imageWidth * 2, frame.type());
		//cout << "Is the shuttlecock launched? (y/n)" << endl;

		cap >> frame;

		//cout << "frame type: " << frame.type() << endl;
		//double T0 = (double)getTickCount();//开始时间
		//double t_passed = 0.0;
		cvtColor(frame, gray_frame, CV_BGR2GRAY);

		if (frame.empty())
			break;
		else {
			count++;
			grayImageL = gray_frame(Rect(0, 0, imageSize.width, imageSize.height));
			grayImageR = gray_frame(Rect(imageSize.width, 0, imageSize.width, imageSize.height));
			//cout << "grayImageL type: " << grayImageL.type() << endl;
				//=================================图像矫正相关操作==================================

			remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_NEAREST);
			remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_NEAREST);
			//t_passed = (double)getTickCount() - T0;//代码运行时间=结束时间-开始时间
			//printf("DRAW===================------execution time = %gms\n", t_passed*1000. / getTickFrequency());//转换时间单位并输出代码运行时间

			//cout << "present frame =" << count << endl;
				//=================================图像矫正相关操作==================================

				//=================================标示画面中有没有飞行的球==================================
			double T0 = (double)getTickCount();//开始时间
			double t_passed = 0.0;
			keykey = 0;
			if (ball_in_air == 0)
			{
				if (count >= 2)
				{
					if (count == 2)
						count += jumpTo;
					int Left_launch = 0, Right_launch = 0;
					resultL = MoveDetect(backgroundL, rectifyImageL, roi_launchL, 0);
					if (isDetected == 1)
					{
						Left_launch = 1;
					}
					resultR = MoveDetect(backgroundR, rectifyImageR, roi_launchR, 1);

					if (isDetected == 1)
					{
						Right_launch = 1;
					}
					//cout << "LEFT RIGHT LAUNCH: " << Left_launch << ", " << Right_launch << endl;
					if (Left_launch == 1 && Right_launch == 1)
					{
						OutFile << "1, " << fileSpeed << ", " << fileAngle << endl;
						cout << "1, " << fileSpeed << ", " << fileAngle << endl;
						if (Upup == 1)
							fileAngle++;
						else if (Upup == 0)
							fileAngle--;
						ball_in_air = 1;
						in_air_count++;
					}
				}
			}
			else if (ball_in_air == 1)
				in_air_count++;

			//t_passed = (double)getTickCount() - T0;//代码运行时间=结束时间-开始时间
			//printf("DRAW===================------execution time = %gms\n", t_passed*1000. / getTickFrequency());//转换时间单位并输出代码运行时间
			//=================================标示画面中有没有飞行的球==================================

			if (count == 1)
			{
				backgroundL = rectifyImageL.clone();
				backgroundR = rectifyImageR.clone();
				roi_launchL = box.extract("trackerL", rectifyImageL);   //第一帧用户画框
				roi_launchR = box.extract("trackerR", rectifyImageR);   //第一帧用户画框
			}

			if (ball_in_air == 1)
			{
				cout << "in_air_count = " << in_air_count << endl;
				if (in_air_count == 1)
				{
					resultL = MoveDetect(backgroundL, rectifyImageL, roi_launchL, 0);
					resultR = MoveDetect(backgroundR, rectifyImageR, roi_launchR, 1);
				}
				else if (in_air_count == 2)
				{
					cout << "current location: (" << current_location[0] << ", " << current_location[1] << ", " << current_location[2] << ")" << endl;
					m_neg[0] = m_pos[0] + m_pos[1]*T;
					m_neg[1] = m_pos[1] - btm * m_pos[1] * m_pos[1] * (((m_pos[1]) < 0) ? -1 : ((m_pos[1]) > 0));
					m_neg[2] = m_pos[2] + m_pos[3] *T;
					m_neg[3] = m_pos[3] - btm * m_pos[3] * m_pos[3] * (((m_pos[3]) < 0) ? -1 : ((m_pos[3]) > 0));
					m_neg[4] = m_pos[4] + m_pos[5] * T;
					m_neg[5] = m_pos[5] - gT - btm * m_pos[5] * m_pos[5] * (((m_pos[5]) < 0) ? -1 : ((m_pos[5]) > 0));
					resultL = MoveDetect(backgroundL, rectifyImageL, roi_launchL, 0);
					resultR = MoveDetect(backgroundR, rectifyImageR, roi_launchL, 1);
				}
				else if (in_air_count >= 3)
				{
					cout << "current location: (" << current_location[0] << ", " << current_location[1] << ", " << current_location[2] << ")" << endl;
					//cout << "pre location: (" << pre_location[0] << ", " << pre_location[1] << ", " << pre_location[2] << ")" << endl;
					m_neg[0] = m_pos[0] + m_pos[1] * T;
					m_neg[1] = m_pos[1] - btm * m_pos[1] * m_pos[1] * (((m_pos[1]) < 0) ? -1 : ((m_pos[1]) > 0));
					m_neg[2] = m_pos[2] + m_pos[3] * T;
					m_neg[3] = m_pos[3] - btm * m_pos[3] * m_pos[3] * (((m_pos[3]) < 0) ? -1 : ((m_pos[3]) > 0));
					m_neg[4] = m_pos[4] + m_pos[5] * T;
					m_neg[5] = m_pos[5] - gT - btm * m_pos[5] * m_pos[5] * (((m_pos[5]) < 0) ? -1 : ((m_pos[5]) > 0));
					cout << "m_neg" << m_neg[0] << ", " << m_neg[1] << ", " << m_neg[2] << ", " << m_neg[3] << ", " << m_neg[4] << ", " << m_neg[5] << endl;
					//=============================预测下一帧位置==============================
					//next_location[0] = 2 * current_location[0] - pre_location[0];
					//next_location[1] = 2 * current_location[1] - pre_location[1];
					//next_location[2] = 2 * current_location[2] - pre_location[2];
					next_location[0] = 4135.71 - m_neg[0] * 1000;
					next_location[1] = m_neg[4] * 1000 - 1200;
					next_location[2] = 7638.09 - m_neg[2] * 1000;
					//predict_next_location(pre_location[0], pre_location[1], pre_location[2], current_location[0], current_location[1], current_location[2],0.033); //方法二
					cout << "next location: (" << m_neg[0] << ", " << m_neg[2] << ", " << m_neg[4] << ")" << endl;
					//=============================预测下一帧位置==============================
					//xl = (roi.x + roi.width / 2) * pixel_length - centerL_x;
					//yl = centerL_y - (roi.y + roi.height / 2) * pixel_length;
					next_xl = (next_location[0] * focal_length / next_location[2] + centerL_x) / pixel_length;   //世界坐标转图像坐标
					next_xr = (focal_length * (next_location[0] - baseline) / next_location[2] + centerR_x) / pixel_length;
					next_yl = (-focal_length * next_location[1] / next_location[2] + centerL_y) / pixel_length;
					next_yr = next_yl;
					int ROI_increase_size = 60;
					predict_roi_L = cvRect(min(1280, max(0, int(next_xl - (predict_box_width + lost_countL * ROI_increase_size) / 2))), min(960, max(0, int(next_yl - (predict_box_height + lost_countL * ROI_increase_size) / 2))), min(1280 - min(1280, max(0, int(next_xl - (predict_box_width + lost_countL * ROI_increase_size) / 2))), int(predict_box_width + lost_countL * ROI_increase_size)), min(960 - min(960, max(0, int(next_yl - (predict_box_height + lost_countL * ROI_increase_size) / 2))), int(predict_box_height + lost_countL * ROI_increase_size)));
					predict_roi_R = cvRect(min(1280, max(0, int(next_xr - (predict_box_width + lost_countR * ROI_increase_size) / 2))), min(960, max(0, int(next_yr - (predict_box_height + lost_countR * ROI_increase_size) / 2))), min(1280 - min(1280, max(0, int(next_xr - (predict_box_width + lost_countR * ROI_increase_size) / 2))), int(predict_box_width + lost_countR * ROI_increase_size)), min(960 - min(960, max(0, int(next_yr - (predict_box_height + lost_countR * ROI_increase_size) / 2))), int(predict_box_height + lost_countR * ROI_increase_size)));
					//cout << "next_location: " << next_location[0] << ", " << next_location[1] << ", " << next_location[2] << endl;
					//cout << "roiXXXXXXXXLLX: " << predict_roi_L.x << ", " << predict_roi_L.y << ", " << predict_roi_L.width << ", " << predict_roi_L.height << endl;
					//cout << "roiXXXXXXXRRRX: " << predict_roi_R.x << ", " << predict_roi_R.y << ", " << predict_roi_R.width << ", " << predict_roi_R.height << endl;
					//double T0 = (double)getTickCount();//开始时间
					//double t_passed = 0.0;
					resultL = MoveDetect(backgroundL, rectifyImageL, predict_roi_L, 0);
					resultR = MoveDetect(backgroundR, rectifyImageR, predict_roi_R, 1);
					//t_passed = (double)getTickCount() - T0;//代码运行时间=结束时间-开始时间
					//printf("DRAW===================------execution time = %gms\n", t_passed*1000. / getTickFrequency());//转换时间单位并输出代码运行时间
					if (last_is_predictedL == 1 || last_is_predictedR == 1)
					{
						xl = next_xl * pixel_length - centerL_x;
						yl = centerL_y - next_yl * pixel_length;
						xr = next_xr * pixel_length - centerR_x;
						yr = centerR_y - next_yr * pixel_length;
					}
				}
				target_z = focal_length * baseline / abs(xl - xr);   //图像坐标转世界坐标
				target_xp = xl * target_z / focal_length;
				//target_xpp = baseline + xr * target_z / focal_length;
				target_y = 0.5*(yl + yr)*target_z / focal_length;
				pre_location[0] = current_location[0]; pre_location[1] = current_location[1]; pre_location[2] = current_location[2];  //保存当前位置和前一帧位置，用于预测下一帧位置
				current_location[0] = (4135.71 - target_xp) / 1000; current_location[1] = (7638.09 - target_z) / 1000; current_location[2] = (1200 + target_y) / 1000;   //保存当前位置和前一帧位置，用于预测下一帧位置
				if (in_air_count == 1)
				{
					cout << "fileSpeed = " << fileSpeed << " fileAngle = " << fileAngle << endl;
					m_pos[0] = current_location[0]; m_pos[2] = current_location[1]; m_pos[4] = current_location[2];
					m_pos[1] = fileSpeed / 3.6*cos((fileAngle * 3 - 50)*3.14 / 180)*cos(0.588);
					m_pos[3] = fileSpeed / 3.6*cos((fileAngle * 3 - 50)*3.14 / 180)*sin(0.588);
					m_pos[5] = fileSpeed / 3.6*sin((fileAngle * 3 - 50)*3.14 / 180);
					cout << "speed_xyz = " << m_pos[1]  << ", " << m_pos[3] << ", " << m_pos[5] << endl;
				}
				else
				{	//cout << "location of shuttlecock: (" << target_xp << ", " << target_y << ", " << target_z << ")" << endl;
					if (sqrt((current_location[0] - m_neg[0])*(current_location[0] - m_neg[0]) + (current_location[1] - m_neg[2])*(current_location[1] - m_neg[2]) + (current_location[2] - m_neg[4])*(current_location[2] - m_neg[4])) >= 1000)
					{
						cout << "K_small" << endl;
						m_pos[0] = m_neg[0] + K_small[0] * (current_location[0] - m_neg[0]);
						m_pos[1] = m_neg[1] + K_small[1] * (current_location[0] - m_neg[0]);
						m_pos[2] = m_neg[2] + K_small[2] * (current_location[1] - m_neg[2]);
						m_pos[3] = m_neg[3] + K_small[3] * (current_location[1] - m_neg[2]);
						m_pos[4] = m_neg[4] + K_small[4] * (current_location[2] - m_neg[4]);
						m_pos[5] = m_neg[5] + K_small[5] * (current_location[2] - m_neg[4]);
					}
					else
					{
						cout << "K_large" << endl;
						m_pos[0] = m_neg[0] + K_large[0] * (current_location[0] - m_neg[0]);
						m_pos[1] = m_neg[1] + K_large[1] * (current_location[0] - m_neg[0]);
						m_pos[2] = m_neg[2] + K_large[2] * (current_location[1] - m_neg[2]);
						m_pos[3] = m_neg[3] + K_large[3] * (current_location[1] - m_neg[2]);
						m_pos[4] = m_neg[4] + K_large[4] * (current_location[2] - m_neg[4]);
						m_pos[5] = m_neg[5] + K_large[5] * (current_location[2] - m_neg[4]);
					}
				}
				cout << "Speed = " << sqrt(m_pos[1]*m_pos[1] + m_pos[3]*m_pos[3] + m_pos[5]*m_pos[5]) << endl;
				//球快着地了 reset
				if (target_y < -1000)
				{
					target_vel = sqrt(pow(current_location[0] - pre_location[0], 2) + pow(current_location[1] - pre_location[1], 2) + pow(current_location[2] - pre_location[2], 2)) / 33;
					//cout << "target_velocity: " << target_vel << "m/s" << endl;
					ball_in_air = 0;
					in_air_count = 0;
					isLastPoint = 1;
				}
				string RRR = "(";
				RRR += to_string(int(target_xp));
				RRR += ", ";
				RRR += to_string(int(target_y));
				RRR += ", ";
				RRR += to_string(int(target_z));
				RRR += ")";
				if (in_air_count <= 2 && in_air_count >= 1)
					putText(resultR, RRR, cvPoint(roi_launchR.x + roi_launchR.width, roi_launchR.y + roi_launchR.height), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 4, 8);
				else if (in_air_count >= 3 || isLastPoint == 1)
					putText(resultR, RRR, cvPoint(predict_roi_R.x + predict_roi_R.width, predict_roi_R.y + predict_roi_R.height), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 4, 8);
				if ((last_is_predictedL || last_is_predictedR) && in_air_count >= 3)
					putText(resultR, "WARNING!Undetected!", cvPoint(predict_roi_R.x + predict_roi_R.width, predict_roi_R.y + predict_roi_R.height - 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 4, 8);
				else if ((last_is_predictedL || last_is_predictedR) && (in_air_count == 1 || in_air_count == 2))
					putText(resultR, "WARNING!Undetected!", cvPoint(roi_launchR.x + roi_launchR.width, roi_launchR.y + roi_launchR.height - 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 23, 0), 4, 8);

			}
			else
			{
				resultL = rectifyImageL;
				resultR = rectifyImageR;
			}

			backgroundL = rectifyImageL.clone();
			backgroundR = rectifyImageR.clone();

			cout << "=====================" << endl;
			//t_passed = (double)getTickCount() - T0;//代码运行时间=结束时间-开始时间
			//printf("DRAW===================------execution time = %gms\n", t_passed*1000. / getTickFrequency());//转换时间单位并输出代码运行时间


			//cout << "result size" << resultL.size() << endl;
			imshow("result_L", resultL);//检测后
			imshow("result_R", resultR);//检测后

			//cout << "press esc to continue" << endl;
			keykey = waitKey(0);
			if (keykey == 27)
			{
				if (in_air_count >= 1 || isLastPoint == 1)
				{
					if (last_is_predictedL || last_is_predictedR)
					{
						OutFile << "0, " << target_xp << ", " << target_y << ", " << target_z << ", 0, " << count << endl;
						cout << "0, " << target_xp << ", " << target_y << ", " << target_z << ", 0";
					}
					else
					{
						OutFile << "0, " << target_xp << ", " << target_y << ", " << target_z << ", 1, " << count << endl;
						cout << "0, " << target_xp << ", " << target_y << ", " << target_z << ", 1";
					}
					cout << " ,present frame =" << count << endl;
					isLastPoint = 0;

				}

				continue;
			}
			else if (keykey == 112) //98:B, 112:P
			{
				do
				{
					roi_nextL = box.extract("trackerL", resultL);
					//circle(resultL, Point(roi_nextL.x, roi_nextL.y), 3, Scalar(0, 0, 255), -1, 8);
					roi_nextR = box.extract("trackerR", resultR);
					//circle(resultR, Point(roi_nextR.x, roi_nextR.y), 3, Scalar(0, 0, 255), -1, 8);
					xl = roi_nextL.x * pixel_length - centerL_x, xr = roi_nextR.x * pixel_length - centerR_x;
					yl = -roi_nextL.y * pixel_length + centerL_y, yr = -roi_nextR.y * pixel_length + centerR_y;
					target_z = focal_length * baseline / abs(xl - xr);
					target_xp = xl * target_z / focal_length;
					//target_xpp = baseline + xr * target_z / focal_length;
					target_y = 0.5*(yl + yr)*target_z / focal_length;
					current_location[0] = (4135.71 - target_xp)/1000; current_location[1] = (7638.09 - target_z)/1000; current_location[2] = (1200 + target_y)/1000;
					//OutFile << "F" << endl;
					//cout << "F";
					//cout << " ,present frame =" << count << endl;
					//OutFile << "0, " << target_xp << ", " << target_y << ", " << target_z << ", ";
					cout << "0, " << target_xp << ", " << target_y << ", " << target_z << ", " << endl;
					cout << "If satisfied, press Esc, if not press any other key to reselect..." << endl;
				} while (waitKey(0) != 27);
				OutFile << "0, " << target_xp << ", " << target_y << ", " << target_z << ", 0, " << count << endl;
				cout << "0, " << target_xp << ", " << target_y << ", " << target_z << ", 0";
				cout << " ,present frame =" << count << endl;
			}
			else if (keykey == 116) //116:T
			{
				ball_in_air = 0;
				in_air_count = 0;
				if (Upup == 1)
					fileAngle--;
				else if (Upup == 0)
					fileAngle++;
			}
		}
	}
	cap.release();
}

void BoxExtractor::mouseHandler(int event, int x, int y, int flags, void *param) {
	BoxExtractor *self = static_cast<BoxExtractor*>(param);
	self->opencv_mouse_callback(event, x, y, flags, param);
}

void BoxExtractor::opencv_mouse_callback(int event, int x, int y, int, void *param) {
	handlerT * data = (handlerT*)param;
	switch (event) {
		// update the selected bounding box
	case EVENT_MOUSEMOVE:
		if (data->isDrawing) {
			data->box.width = x - data->box.x;
			data->box.height = y - data->box.y;
		}
		break;

		// start to select the bounding box
	case EVENT_LBUTTONDOWN:
		data->isDrawing = true;
		data->box = cvRect(x, y, 0, 0);
		break;

		// cleaning up the selected bounding box
	case EVENT_LBUTTONUP:
		data->isDrawing = false;
		if (data->box.width < 0) {
			data->box.x += data->box.width;
			data->box.width *= -1;
		}
		if (data->box.height < 0) {
			data->box.y += data->box.height;
			data->box.height *= -1;
		}
		break;
	}
}

Rect2d BoxExtractor::extract(Mat img) {
	return extract("Bounding Box Extractor", img);
}

Rect2d BoxExtractor::extract(const std::string& windowName, Mat img, bool showCrossair) {

	int key = 0;

	// show the image and give feedback to user
	imshow(windowName, img);
	printf("Select an object to track and then press SPACE/BACKSPACE/ENTER button!\n");

	// copy the data, rectangle should be drawn in the fresh image
	params.image = img.clone();

	// select the object
	setMouseCallback(windowName, mouseHandler, (void *)&params);

	// end selection process on SPACE (32) BACKSPACE (27) or ENTER (13)
	while (!(key == 32 || key == 27 || key == 13)) {
		// draw the selected object
		rectangle(
			params.image,
			params.box,
			Scalar(255, 0, 0), 2, 1
		);

		// draw cross air in the middle of bounding box
		if (showCrossair) {
			// horizontal line
			line(
				params.image,
				Point((int)params.box.x, (int)(params.box.y + params.box.height / 2)),
				Point((int)(params.box.x + params.box.width), (int)(params.box.y + params.box.height / 2)),
				Scalar(255, 0, 0), 2, 1
			);

			// vertical line
			line(
				params.image,
				Point((int)(params.box.x + params.box.width / 2), (int)params.box.y),
				Point((int)(params.box.x + params.box.width / 2), (int)(params.box.y + params.box.height)),
				Scalar(255, 0, 0), 2, 1
			);
		}

		// show the image bouding box
		imshow(windowName, params.image);

		// reset the image
		params.image = img.clone();

		//get keyboard event
		key = waitKey(1);
	}


	return params.box;
}
