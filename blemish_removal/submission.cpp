#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int radius = 15;

void putImage(Mat & baseMat, const Mat & newMat, int col, int row ){
    int side = newMat.size().width;
    newMat.copyTo(baseMat(Range(side * col, side * (col+1)),Range(side * row, side * (row+1) )));
}

void thresoldChannels(const Mat &roi) {
    Mat hsv_cv;
    cvtColor(roi, hsv_cv, COLOR_BGR2HSV);

    Mat hsv_cv_channels[3], hsv_cv_masks[3];
    Mat rgb_cv_channels[3], rgb_cv_masks[3];

    split(hsv_cv, hsv_cv_channels);
    split(roi, rgb_cv_channels);

    Mat blemishRes = Mat::zeros(roi.size().height*3, roi.size().width*3, CV_8UC1);

    for (int i=0; i<3; i++ ) {
        threshold(hsv_cv_channels[i], hsv_cv_masks[i], mean(hsv_cv_channels[i])[0]-10, 255, ThresholdTypes::THRESH_BINARY_INV );
        threshold(rgb_cv_channels[i], rgb_cv_masks[i], mean(rgb_cv_channels[i])[0]-10, 255, ThresholdTypes::THRESH_BINARY_INV );

        putImage(blemishRes, hsv_cv_masks[i], 0, i);
        putImage(blemishRes, rgb_cv_masks[i], 1, i);
    }

    Mat roi_gray, roi_grayMask;
    cvtColor(roi, roi_gray, COLOR_BGR2GRAY);
    threshold(roi_gray, roi_grayMask, mean(roi_gray)[0]-10, 255, ThresholdTypes::THRESH_BINARY_INV );

    putImage(blemishRes, roi_grayMask, 2, 1);

    imshow("Thresholded channels", blemishRes);
}

int main(){
    Mat source = imread("../images/blemish.png");

    if (source.empty())
    {
        cout << "Unable to read file" << endl;
        return 2;
    }

    namedWindow("Window");

    // set up lambda callback
    cv::setMouseCallback( "Window",
        [] (int event, int x, int y, int flags, void* data) {

        if( event == EVENT_LBUTTONDOWN )
        {
            Mat image = *(Mat*)data;

            Rect r (x-radius, y-radius, radius*2,radius*2);

            // crop rectangle area
            Mat roi = image(r);

            // lets find out the best channel for work
            thresoldChannels(roi);

            // remove blemish
            // using green channel
            Mat rgb_channels[3];
            Mat green_mask;
            split(roi, rgb_channels);

            threshold(rgb_channels[1], green_mask, mean(rgb_channels[1])[0]-10, 255, ThresholdTypes::THRESH_BINARY);

            erode(green_mask, green_mask, getStructuringElement(MORPH_ELLIPSE, Size(3,3)));

            Mat maskChannels[] = {green_mask,green_mask,green_mask};
            green_mask = green_mask/255;

            Mat imageChannels[3];
            split(roi, imageChannels);

            Mat fixedImage[3];

            Mat meanImage(roi.size().width, roi.size().height, CV_8UC3, mean(roi));
            Mat meanImageChannels[3];
            split(meanImage, meanImageChannels);

            for (int i = 0; i < 3; i++) {
                multiply(imageChannels[i], maskChannels[i], imageChannels[i]);
                multiply(meanImageChannels[i], (1-maskChannels[i]), meanImageChannels[i]);

                bitwise_or(imageChannels[i], meanImageChannels[i], fixedImage[i] );
            }

            Mat removedBlemishImage, fixedBlemish ;
            merge(imageChannels, 3, removedBlemishImage);
            merge(meanImageChannels, 3, meanImage);
            merge(fixedImage, 3, fixedBlemish);

            Mat res = Mat::zeros(roi.size().width, roi.size().height*3, CV_8UC3);

            putImage(res, removedBlemishImage, 0, 0);
            putImage(res, meanImage, 0, 1);
            putImage(res, fixedBlemish, 0, 2);

            imshow("Image", res);

            // replace fixed roi on original image with the mask

            // crop circle area
            cv::Mat circleMask = cv::Mat::zeros( roi.rows, roi.cols, CV_8UC1 );
            circle( circleMask, Point(roi.size().width/2, roi.size().width/2), radius, Scalar(255,255,255), -1);

            fixedBlemish.copyTo(image(r), circleMask);
        }
    }, &source );

    while(true)
    {
        imshow("Window", source );

        int k = waitKey(20) & 0xFF;
        if ( k == 27 || k == 'q' )
        {
            break;
        }
    }
    return 0;
}
