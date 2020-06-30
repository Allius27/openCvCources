
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat convertBGRtoGray(Mat image){
    ///
    /// YOUR CODE HERE

    if (image.empty())
        return Mat::zeros(0, 0, CV_8UC1);

    Mat bgr_array[3];
    Mat result = Mat::zeros(image.size(), CV_8UC1);

    split(image, bgr_array);

    result = bgr_array[0] * 0.114 + bgr_array[1] * 0.587 + bgr_array[2] * 0.299;

    return result;
    ///
}


Mat convertBGRtoHSV(Mat image){
    ///
    /// YOUR CODE HERE

    cv::Mat hsvImage = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC3);
    double h, s, v, b, g, r, min_value, max_value;

    for(int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            b = (static_cast<double>(image.at<cv::Vec3b>(i, j)[0])) / 255;
            g = (static_cast<double>(image.at<cv::Vec3b>(i, j)[1])) / 255;
            r = (static_cast<double>(image.at<cv::Vec3b>(i, j)[2])) / 255;

            max_value = max(max(r,g),b);
            min_value = min(min(r,g),b);

            v = max_value;

            if (v == 0) {
                s = 0;
            } else {
                s = (v - min_value) / v;
            }

            if (v == r) {
                h = (60 * (g - b)) / (v - min_value);
            } else if (v == g) {
                h = 120 + (60 * (b - r)) / (v - min_value);
            } else if (v == b) {
                h = 240 + (60 * (r - g)) / (v - min_value);
            }

            if (h < 0) {
                h += 360;
            }

            hsvImage.at<cv::Vec3b>(i, j)[2]= v * 255;
            hsvImage.at<cv::Vec3b>(i, j)[1] = s * 255;
            hsvImage.at<cv::Vec3b>(i, j)[0] = h / 2;
        }
    }

    return hsvImage;

    ///
}


int main(){

    auto printText = [&](Mat & mat, const string &text) {
        putText(mat, text, Point(10, 50), FONT_HERSHEY_PLAIN, 2, (0,0,0));
    };

    Mat image = imread("../images/Dolores.jpg");

    if (image.empty())
        return EXIT_FAILURE;

    resize(image, image, Size(), 0.4, 0.4);
    Mat resultMat = Mat::zeros(image.rows*2, image.cols*2, CV_8UC3);

    Mat gray_custom = convertBGRtoGray(image);
    printText(gray_custom, "custom_gray");
    cvtColor(gray_custom,gray_custom, COLOR_GRAY2BGR);
    gray_custom.copyTo(resultMat(Range(0,image.size().height),Range(0,image.size().width)));

    Mat gray_cv;
    cvtColor(image, gray_cv, COLOR_BGR2GRAY);
    printText(gray_cv, "opencv_gray");
    cvtColor(gray_cv,gray_cv, COLOR_GRAY2BGR);
    gray_cv.copyTo(resultMat(Range(0, image.size().height),Range(image.size().width,image.size().width*2)));

    Mat hsv_cv;
    cvtColor(image, hsv_cv, COLOR_BGR2HSV);
    printText(hsv_cv, "opencv_hsv");
    hsv_cv.copyTo(resultMat(Range(image.size().height, image.size().height*2),Range(0,image.size().width)));

    Mat hsv_custom = convertBGRtoHSV(image);
    printText(hsv_custom, "custom_hsv");
    hsv_custom.copyTo(resultMat(Range(image.size().height, image.size().height*2),Range(image.size().width,image.size().width*2)));

    imshow("resultMat",resultMat);
    waitKey();

    return EXIT_SUCCESS;
}
