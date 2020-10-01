#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;


void putImage(Mat & baseMat, const Mat & newMat, int col, int row ){
    int width = newMat.size().width;
    int height = newMat.size().height;

    newMat.copyTo(baseMat(Range(height * col, height * (col+1)),
                          Range(width  * row, width  * (row+1) )));
}

void thresoldChannels(const Mat &roi) {
    Mat hsv_cv;
    cvtColor(roi, hsv_cv, COLOR_BGR2HSV);

    Mat hsv_cv_channels[3], hsv_cv_masks[3];
    Mat rgb_cv_channels[3], rgb_cv_masks[3];

    split(hsv_cv, hsv_cv_channels);
    split(roi, rgb_cv_channels);

    Mat channelsRes = Mat::zeros(roi.size().height*3, roi.size().width*3, CV_8UC1);

    for (int i=0; i<3; i++ ) {
        threshold(hsv_cv_channels[i], hsv_cv_masks[i], mean(hsv_cv_channels[i])[0], 255, ThresholdTypes::THRESH_BINARY_INV );
        threshold(rgb_cv_channels[i], rgb_cv_masks[i], mean(rgb_cv_channels[i])[0], 255, ThresholdTypes::THRESH_BINARY_INV );

        putImage(channelsRes, hsv_cv_masks[i], 0, i);
        putImage(channelsRes, rgb_cv_masks[i], 1, i);
    }

    Mat roi_gray, roi_grayMask;
    cvtColor(roi, roi_gray, COLOR_BGR2GRAY);
    threshold(roi_gray, roi_grayMask, mean(roi_gray)[0], 255, ThresholdTypes::THRESH_BINARY_INV );

    putImage(channelsRes, roi_grayMask, 2, 1);

    imshow("channels", channelsRes);
    waitKey(0);
}


float estimateOverlight(Mat & image, const Mat & mask){
//!  simple thresholding shows, that we might use value channel from hsv model
    thresoldChannels(image);

    Mat hsv, hsvChannels[3], valueMask;

    cvtColor(image, hsv, COLOR_BGR2HSV);
    split(hsv, hsvChannels);

    int _mean = mean(hsvChannels[1])[0]-15;
    threshold(hsvChannels[1], valueMask, _mean, 255, ThresholdTypes::THRESH_BINARY_INV );
    bitwise_and(valueMask, mask, valueMask);

    auto lightAmount = [](const cv::Mat &mat) {
        int nonZeroValues{0};

        for(int h = 0; h < mat.size().height; h++)
            for(int w = 0; w < mat.size().width; w++)
                if ( (int)(mat.at<uchar>(h,w)) != 0 )
                    nonZeroValues++;

        return nonZeroValues;
    };

    int maskAmount = lightAmount(mask);
    int valueAmount = lightAmount(valueMask);

    cvtColor(valueMask, image, COLOR_GRAY2BGR);

    return (float)(valueAmount * (float)100/maskAmount);
}


int main() {

    Mat faceNorm = imread("../images/face_normal.jpg");
    Mat faceOver = imread("../images/face_overlight.jpg");

    if ( faceNorm.empty() || faceOver.empty() ) {
        cerr << "Some of images is empty" << endl;
        exit(2);
    }

    vector <Point> landmarks = {{210,180}, {195,235}, {200,300}, {250,360},
                                {300,360}, {363,300}, {368,235}, {353,180} };

    vector<int> vecX,vecY;

    for ( Point p : landmarks ) {
        vecX.push_back(p.x);
        vecY.push_back(p.y);
    }

    Rect rect;
    rect.x = *std::min_element(vecX.begin(), vecX.end());
    rect.width = *std::max_element(vecX.begin(), vecX.end()) - rect.x;
    rect.y = *std::min_element(vecY.begin(), vecY.end());
    rect.height = *std::max_element(vecY.begin(), vecY.end()) - rect.y;

    Mat mask = cv::Mat::zeros(faceNorm.rows, faceNorm.cols, CV_8UC1);
    fillConvexPoly(mask, landmarks.data(), landmarks.size(), cv::Scalar(255));

    Mat roiNorm, roiOver;

    faceNorm.copyTo(roiNorm, mask);
    faceOver.copyTo(roiOver, mask);
    roiNorm = roiNorm(rect);
    roiOver = roiOver(rect);

    for ( auto p: landmarks ) {
        circle(faceNorm, p, 2, (255,255,0), 2);
        circle(faceOver, p, 2, (255,255,0), 2);
    }

    float normEstimate = estimateOverlight(roiNorm, mask(rect));
    float overEstimate = estimateOverlight(roiOver, mask(rect));

    Mat showImage = Mat::zeros(faceNorm.size().height, faceNorm.size().width*2 + rect.width, CV_8UC3);

    putImage(showImage, faceNorm, 0, 0);
    putImage(showImage, faceOver, 0, 1);

    roiNorm.copyTo(showImage(Rect(faceNorm.size().width*2,
                                  faceNorm.size().height/2 - rect.height,
                                  rect.width,
                                  rect.height)));

    roiOver.copyTo(showImage(Rect(faceNorm.size().width*2,
                                  faceNorm.size().height/2,
                                  rect.width,
                                  rect.height)));

    putText(showImage, to_string(normEstimate), Point(20,50), FONT_HERSHEY_COMPLEX, 1.0, Scalar(255,255,0) );
    putText(showImage, to_string(overEstimate), Point(690,50), FONT_HERSHEY_COMPLEX, 1.0, Scalar(255,255,0) );

    imshow("showImage", showImage);
    waitKey(0);
}
