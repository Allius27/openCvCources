#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

Mat makeHistogram(const Mat & src) {

    /// Separate the image in 3 places ( B, G and R )
    vector<Mat> bgr_planes;
    split( src, bgr_planes );

    /// Establish the number of bins
    int histSize = 256;

    /// Set the ranges ( for B,G,R) )
    float range[] = { 0, 256 } ;
    const float* histRange = { range };

    bool uniform = true; bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    /// Compute the histograms:
    calcHist( &bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate );
    calcHist( &bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate );

    b_hist.at<float>(0,0) = 0;
    g_hist.at<float>(0,0) = 0;
    r_hist.at<float>(0,0) = 0;

    vector <int> hst;

    for(int h = 0; h < b_hist.size().height; h++)
        hst.push_back((int)b_hist.at<float>(h,0));

    struct quality {
        float black = 0.f;
        int blackAmount = 0;
        bool isOverBlack = false;

        float semitone = 0.f;
        int semitoneAmount = 0;

        float light = 0.f;
        int lightAmount = 0;
        bool isOverLight = false;
    };

    auto estimateQuality = [](const vector<int>& vec){
        quality _quality;

        auto getRatio = [](const vector<int>& vec, int minPos, int maxPos){
            int maxLocalValue  = (int)( *std::max_element(vec.begin()+minPos, vec.begin()+maxPos));
            int maxGlobalValue = (int)( *std::max_element(vec.begin(), vec.end()));

            return (float)maxLocalValue/maxGlobalValue;
        };

        auto getAmount = [](const vector<int>& vec, int minPos, int maxPos){

            int amount = 0;

            for (int i=minPos; i<maxPos; i++ ) {
                amount += vec.at(i);
            }

            return amount;
        };

        // estimate black area
        _quality.black = getRatio(vec, 0, vec.size()/3);
        _quality.blackAmount = getAmount(vec, 0, vec.size()/3);

        // estimate semitone area
        _quality.semitone = getRatio(vec, vec.size()/3, vec.size()/3*2);
        _quality.semitoneAmount = getAmount(vec, vec.size()/3, vec.size()/3*2);

        // estimate white area
        _quality.light = getRatio(vec, vec.size()/3*2, vec.size());
        _quality.lightAmount = getAmount(vec, vec.size()/3*2, vec.size());

        if (_quality.blackAmount > _quality.semitoneAmount)
            _quality.isOverBlack = true;

        if (_quality.lightAmount > _quality.semitoneAmount)
            _quality.isOverLight = true;

        return _quality;
    };

    quality _q = estimateQuality(hst);

    cout << " semitone ratio: " << _q.semitone << " amount: " << _q.semitoneAmount
        << "black ratio: "     << _q.black << " amount: " << _q.blackAmount << " isOver: " << _q.isOverBlack
         << " light ratio: "    << _q.light << " amount: " << _q.lightAmount << " isOver: " << _q.isOverLight
         << endl;

    // Draw the histograms for B, G and R
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );

    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

    /// Normalize the result to [ 0, histImage.rows ]
    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

    /// Draw for each channel
    for( int i = 1; i < histSize; i++ )
    {
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(b_hist.at<float>(i-1)) ) ,
                         Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
                         Scalar( 255, 0, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(g_hist.at<float>(i-1)) ) ,
                         Point( bin_w*(i), hist_h - cvRound(g_hist.at<float>(i)) ),
                         Scalar( 0, 255, 0), 2, 8, 0  );
        line( histImage, Point( bin_w*(i-1), hist_h - cvRound(r_hist.at<float>(i-1)) ) ,
                         Point( bin_w*(i), hist_h - cvRound(r_hist.at<float>(i)) ),
                         Scalar( 0, 0, 255), 2, 8, 0  );
    }

    line (histImage, Point(histImage.size().width/3, 0), Point(histImage.size().width/3, histImage.size().height), {255,0,0});
    line (histImage, Point(histImage.size().width/3*2, 0), Point(histImage.size().width/3*2, histImage.size().height), {255,0,0});

    rectangle(histImage, Rect(0,0,histImage.size().width, histImage.size().height), Scalar(0,255,0));

    return histImage;
}


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
}


float estimateOverlight(Mat & image, const Mat & mask){
//!  simple thresholding shows, that we might use value channel from hsv model
    thresoldChannels(image);

    Mat hsv, hsvChannels[3], valueMask;

    cvtColor(image, hsv, COLOR_BGR2HSV);
    split(hsv, hsvChannels);

//    int _mean = mean(hsvChannels[1])[0];
    int _mean = 42;
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

    Rect roiRect;
    roiRect.x = *std::min_element(vecX.begin(), vecX.end());
    roiRect.width = *std::max_element(vecX.begin(), vecX.end()) - roiRect.x;
    roiRect.y = *std::min_element(vecY.begin(), vecY.end());
    roiRect.height = *std::max_element(vecY.begin(), vecY.end()) - roiRect.y;

    Mat mask = cv::Mat::zeros(faceNorm.rows, faceNorm.cols, CV_8UC1);
    fillConvexPoly(mask, landmarks.data(), landmarks.size(), cv::Scalar(255));

    Mat roiNorm, roiOver;

    faceNorm.copyTo(roiNorm, mask);
    faceOver.copyTo(roiOver, mask);
    roiNorm = roiNorm(roiRect);
    roiOver = roiOver(roiRect);

    for ( auto p: landmarks ) {
        circle(faceNorm, p, 2, (255,255,0), 2);
        circle(faceOver, p, 2, (255,255,0), 2);
    }

    Mat grayNorm, grayOver;

    cvtColor(roiNorm,grayNorm, COLOR_BGR2GRAY); cvtColor(grayNorm,grayNorm, COLOR_GRAY2BGR);
    cvtColor(roiOver,grayOver, COLOR_BGR2GRAY); cvtColor(grayOver,grayOver, COLOR_GRAY2BGR);

    Mat histNorm = makeHistogram(grayNorm);
    Mat histOver = makeHistogram(grayOver);

    float normEstimate = estimateOverlight(roiNorm, mask(roiRect));
    float overEstimate = estimateOverlight(roiOver, mask(roiRect));

    Mat showImage = Mat::zeros(faceNorm.size().height+histOver.size().height , faceNorm.size().width*2 + roiRect.width*2, CV_8UC3);

    faceNorm.copyTo(showImage(Rect(0, 0, faceNorm.size().width, faceNorm.size().height)));
    faceOver.copyTo(showImage(Rect(faceOver.size().width+roiRect.width, 0, faceOver.size().width, faceOver.size().height)));

    roiNorm.copyTo(showImage(Rect(faceNorm.size().width,
                                  (faceNorm.size().height - roiRect.height)/2,
                                  roiRect.width,
                                  roiRect.height)));

    roiOver.copyTo(showImage(Rect(faceNorm.size().width*2+roiRect.width,
                                  (faceNorm.size().height - roiRect.height)/2,
                                  roiRect.width,
                                  roiRect.height)));

    histNorm.copyTo(showImage(Rect((faceNorm.size().width - histNorm.size().width)/2,
                                   faceNorm.size().height,
                                   histNorm.size().width,
                                   histNorm.size().height)));

    histOver.copyTo(showImage(Rect(faceNorm.size().width + (faceNorm.size().width - histOver.size().width)/2 + roiRect.width,
                                   faceNorm.size().height,
                                   histOver.size().width,
                                   histOver.size().height)));

    putText(showImage, to_string(normEstimate), Point(20,50), FONT_HERSHEY_COMPLEX, 1.0, Scalar(255,255,0) );
    putText(showImage, to_string(overEstimate), Point(890,50), FONT_HERSHEY_COMPLEX, 1.0, Scalar(255,255,0) );

    resize(showImage,showImage,Size(), 0.8, 0.8);
    imshow("showImage", showImage);
    waitKey(0);
}
