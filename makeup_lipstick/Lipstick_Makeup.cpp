#include <opencv2/opencv.hpp>
#include "faceBlendCommon.hpp"

using namespace std;
using namespace cv;

int main () {
    // Landmark model location
    string PREDICTOR_PATH =  "../resource/lib/publicdata/models/shape_predictor_68_face_landmarks.dat";

    // Get the face detector
    dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();
    // The landmark detector is implemented in the shape_predictor class
    dlib::shape_predictor landmarkDetector;
    dlib::deserialize(PREDICTOR_PATH) >> landmarkDetector;

    Mat img = imread("../images/girl-no-makeup.jpg");

    resize(img,img, Size(), 0.7, 0.7);

    if (img.empty())
    {
        cout << "Unable to read file" << endl;
        return 2;
    }

    vector<Point2f> points = getLandmarks(faceDetector, landmarkDetector, img);

    vector<Point> lipsPoints;
    // copy dots of lips
    for ( int ind = 48; ind <= 59; ind++)
        lipsPoints.push_back(points.at(ind) );

    std::vector<std::vector<cv::Point> > fillPoints {lipsPoints};

    Mat mask = cv::Mat::zeros(img.size().height, img.size().width, CV_8UC1);
    cv::fillPoly(mask, fillPoints, cv::Scalar(255));

    Rect lipsRect = boundingRect(lipsPoints);

    Mat lipsCanvas = img(lipsRect);
    Mat lipsMask = mask(lipsRect);

    Mat lipsMasked;
    lipsCanvas.copyTo(lipsMasked, lipsMask);

    Mat layers[3];
    split(lipsMasked,layers);

    // add intensity of red color
    layers[2] += 30;

    Mat makeUpLips;
    merge(layers, 3,  makeUpLips);

    Mat makeUpImage = img.clone();
    makeUpLips.copyTo(makeUpImage(lipsRect), lipsMask);

    Mat result;
    hconcat(img, makeUpImage, result);

    imshow("Final image", result);
    waitKey(0);
}













